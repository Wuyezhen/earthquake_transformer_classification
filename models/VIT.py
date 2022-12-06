import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer,Dense,LayerNormalization,GlobalAveragePooling1D,Softmax,Dropout,Conv1D
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn
import numpy as np
from tensorflow import einsum
from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, input_resolution, dim, norm_layer=LayerNormalization, prefix=''):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = Dense(2* dim, use_bias=False,
                               name=f'{prefix}/downsample/reduction')
        self.norm = norm_layer(epsilon=1e-5, name=f'{prefix}/downsample/norm')
        print(input_resolution)

    def call(self, x):
        # H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()
        # assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # x = tf.reshape(x, shape=[-1, H, W, C])

        x0 = x[:, 0::4, :]  # B L/4 C
        x1 = x[:, 1::4, :]  # B L/4 C
        x2 = x[:, 2::4, :]  # B L/4 C
        x3 = x[:, 3::4, :]  # B L/4 C

        x = tf.concat([x0, x1, x2, x3], axis=-1)
        # x = tf.reshape(x, shape=[-1, (H // 2) * (W // 2), 4 * C])
        # x = Conv1D(self.dim * 2, kernel_size=4, strides=4, padding='same')(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x

class PreNorm(Layer):
    def __init__(self, fn):
        super(PreNorm, self).__init__()

        self.norm = LayerNormalization()
        self.fn = fn

    def call(self, x, training=True):
        return self.fn(self.norm(x), training=training)

class MLP(Layer):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super(MLP, self).__init__()

        def GELU():
            def gelu(x, approximate=False):
                if approximate:
                    coeff = tf.cast(0.044715, x.dtype)
                    return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
                else:
                    return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))

            return nn.Activation(gelu)

        self.net = Sequential([
            Dense(units=hidden_dim),
            GELU(),
            Dropout(rate=dropout),
            Dense(units=dim),
            Dropout(rate=dropout)
        ])

    def call(self, x, training=True):
        return self.net(x, training=training)

class Attention(Layer):
    def __init__(self, dim,image_size, heads=8):
        super(Attention, self).__init__()
        self.dim = dim
        head_dim = dim // heads
        self.window_size = int(np.sqrt(image_size))
        self.heads = heads
        self.prefix=''
        self.scale = head_dim ** -0.5
        self.attend = Softmax()
        self.to_qkv = Dense(units=dim * 3, use_bias=True, name=f'{self.prefix}/attn/qkv')
        self.proj = Dense(dim, name=f'{self.prefix}/attn/proj')
        self.relative_position_bias_table = self.add_weight(f'{self.prefix}/attn/relative_position_bias_table',
                                                            shape=(
                                                                (2 * self.window_size - 1) * (
                                                                        2 * self.window_size - 1),
                                                                self.heads),
                                                            initializer=tf.initializers.Zeros(), trainable=True)

        coords_h = np.arange(self.window_size)
        coords_w = np.arange(self.window_size)
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :,
                          None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1).astype(np.int64)
        self.relative_position_index = tf.Variable(initial_value=tf.convert_to_tensor(
            relative_position_index), trainable=False,name=f'{self.prefix}/attn/relative_position_index')
        # print('dim:{},head:{},szie:{}')

    def call(self, x, training=True):
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # dots = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2])) * self.scale
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        relative_position_bias = tf.gather(self.relative_position_bias_table, tf.reshape(
            self.relative_position_index, shape=[-1]))
        relative_position_bias = tf.reshape(relative_position_bias, shape=[
            self.window_size * self.window_size, self.window_size * self.window_size, -1])
        relative_position_bias = tf.transpose(
            relative_position_bias, perm=[2, 0, 1])
        dots = dots + tf.expand_dims(relative_position_bias, axis=0)
        attn = self.attend(dots)

        # x = tf.matmul(attn, v)
        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.proj(x)

        return x

class Transformer(Layer):
    def __init__(self, dim, image_size, heads, mlp_dim, downsample=None, dropout=0.0):
        super(Transformer, self).__init__()
        if downsample is not None:
            self.patchmerge = PatchMerging(input_resolution=image_size, dim=dim)
        else:
            self.patchmerge = None
        self.attn = PreNorm(Attention(dim,image_size, heads=heads))
        self.mlp = PreNorm(MLP(dim, dim*mlp_dim, dropout=dropout))


    def call(self, x, training=True):

        x = self.attn(x, training=training) + x
        x = self.mlp(x, training=training) + x
        if self.patchmerge is not None:
            x = self.patchmerge(x)
        return x

class ViT(Model):
    def __init__(self, image_size=32768, patch_size=32, num_classes=2, dim=12, depth=4, heads=[3, 6, 12, 24], mlp_dim=4.,
                dropout=0.0):
        super(ViT, self).__init__()

        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)

        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = image_size // patch_size
        # assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_embedding = Sequential([
            # Rearrange('b (L p) c -> b L (p c)', p=patch_size),
            # Dense(dim),
            Conv1D(dim, kernel_size=patch_size, strides=patch_size, padding='same'),
            LayerNormalization()
        ], name='patch_embedding')

        # self.pos_embedding = tf.Variable(initial_value=tf.random.normal([1, num_patches + 1, dim]))
        # self.cls_token = tf.Variable(initial_value=tf.random.normal([1, 1, dim]))

        self.blocks = tf.keras.Sequential([Transformer(dim=int(dim*2**i), image_size=(image_size//patch_size)//(4**i),
                                                       heads=heads[i], mlp_dim=mlp_dim, dropout=dropout,
                                                       downsample=PatchMerging if (i< depth-1) else None)
                                                       for i in range(depth)])


        # self.pool = pool
        self.mlp_head = Sequential([
            LayerNormalization(epsilon=1e-5),
            GlobalAveragePooling1D(),
            Dense(num_classes, activation='softmax')
        ], name='mlp_head')

    def call(self, img, training=True):
        x = self.patch_embedding(img)

        # x += self.pos_embedding[:, :(n + 1)]
        # x = self.dropout(x, training=training)

        x = self.blocks(x)

        # if self.pool == 'mean':
        #     x = tf.reduce_mean(x, axis=1)
        # else:
        #     x = x[:, 0]

        x = self.mlp_head(x)

        return x

