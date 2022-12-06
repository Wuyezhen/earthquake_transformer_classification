import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Conv1D, LayerNormalization, GlobalAveragePooling1D
from einops import rearrange, repeat

CFGS = {
    'swin_tiny_224': dict(input_size=(224, 224), window_size=7, embed_dim=96, depths=[2, 2, 6, 2],
                          num_heads=[3, 6, 12, 24]),
    'swin_small_224': dict(input_size=(224, 224), window_size=7, embed_dim=96, depths=[2, 2, 18, 2],
                           num_heads=[3, 6, 12, 24]),
    'swin_base_224': dict(input_size=(224, 224), window_size=7, embed_dim=128, depths=[2, 2, 18, 2],
                          num_heads=[4, 8, 16, 32]),
    'swin_base_384': dict(input_size=(384, 384), window_size=12, embed_dim=128, depths=[2, 2, 18, 2],
                          num_heads=[4, 8, 16, 32]),
    'swin_large_224': dict(input_size=(224, 224), window_size=7, embed_dim=192, depths=[2, 2, 18, 2],
                           num_heads=[6, 12, 24, 48]),
    'swin_large_384': dict(input_size=(384, 384), window_size=12, embed_dim=192, depths=[2, 2, 18, 2],
                           num_heads=[6, 12, 24, 48])
}


def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


class Mlp(tf.keras.layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., prefix=''):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Dense(hidden_features, name=f'{prefix}/mlp/fc1')
        self.fc2 = Dense(out_features, name=f'{prefix}/mlp/fc2')
        self.drop = Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, L, C = x.get_shape().as_list()
    x = tf.reshape(x, shape=[-1, L // window_size, window_size, C])
    # x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, shape=[-1, window_size, C])
    return windows


def window_reverse(windows, window_size, L, C):
    x = tf.reshape(windows, shape=[-1, L // window_size, window_size, C])
    # x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, shape=[-1, L, C])
    return x


class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,shuffle=False,
                 prefix=''):
        super().__init__()
        self.dim = dim
        self.window_size = int(np.sqrt(window_size))
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.prefix = prefix
        self.shuffle = shuffle
        self.qkv = Dense(dim * 3, use_bias=qkv_bias,
                         name=f'{self.prefix}/attn/qkv')
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(dim, name=f'{self.prefix}/attn/proj')
        self.proj_drop = Dropout(proj_drop)

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(f'{self.prefix}/attn/relative_position_bias_table',
                                                            shape=(
                                                                (2 * self.window_size - 1) * (
                                                                            2 * self.window_size - 1),
                                                                self.num_heads),
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
            relative_position_index), trainable=False, name=f'{self.prefix}/attn/relative_position_index')
        self.built = True

    def call(self, x, mask=None):
        B_, N, C = x.get_shape().as_list()
        qkv = self.qkv(x)
        if self.shuffle:
            q, k, v = rearrange(qkv, 'b (ws1 ws2 L) (qkv h d) -> qkv (b L) h (ws1 ws2) d', h=self.num_heads, qkv=3,
                                ws1=self.window_size, ws2=self.window_size)
        else:
            q, k, v = rearrange(qkv, 'b (L ws1 ws2) (qkv h d) -> qkv (b L) h (ws1 ws2) d', h=self.num_heads, qkv=3,
                                ws1=self.window_size, ws2=self.window_size)

        q = q * self.scale
        attn = (q @ tf.transpose(k, perm=[0, 1, 3, 2]))
        relative_position_bias = tf.gather(self.relative_position_bias_table, tf.reshape(
            self.relative_position_index, shape=[-1]))
        relative_position_bias = tf.reshape(relative_position_bias, shape=[
            self.window_size * self.window_size, self.window_size * self.window_size, -1])
        relative_position_bias = tf.transpose(
            relative_position_bias, perm=[2, 0, 1])
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]  # tf.shape(mask)[0]
            attn = tf.reshape(attn, shape=[-1, nW, self.num_heads, self.window_size*self.window_size, self.window_size*self.window_size]) + tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), attn.dtype)
            attn = tf.reshape(attn, shape=[-1, self.num_heads, self.window_size*self.window_size, self.window_size*self.window_size])
            attn = tf.nn.softmax(attn, axis=-1)
        else:
            attn = tf.nn.softmax(attn, axis=-1)

        attn = self.attn_drop(attn)

        x = attn @ v
        if self.shuffle:
            x = rearrange(x, '(b L) h (ws1 ws2) d -> b (ws1 ws2 L) (h d)', h=self.num_heads, b=B_, ws1=self.window_size)
        else:
            x = rearrange(x, '(b L) h (ws1 ws2) d -> b (L ws1 ws2) (h d)', h=self.num_heads, b=B_, ws1=self.window_size)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def drop_path(inputs, drop_prob, is_training):
    if (not is_training) or (drop_prob == 0.):
        return inputs

    # Compute keep_prob
    keep_prob = 1.0 - drop_prob

    # Compute drop_connect tensor
    random_tensor = keep_prob
    shape = (tf.shape(inputs)[0],) + (1,) * \
            (len(tf.shape(inputs)) - 1)
    random_tensor += tf.random.uniform(shape, dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * binary_tensor
    return output


class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        return drop_path(x, self.drop_prob, training)


class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size=16, shift_size=0, mlp_ratio=4.,shuffle=False,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path_prob=0., norm_layer=LayerNormalization,
                 prefix=''):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if self.input_resolution <= self.window_size:
            self.shift_size = 0
            self.window_size = int(np.floor(np.sqrt(self.input_resolution))**2)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.prefix = prefix

        self.norm1 = norm_layer(epsilon=1e-5, name=f'{self.prefix}/norm1')
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads,shuffle=shuffle,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                                    prefix=self.prefix)
        self.drop_path = DropPath(
            drop_path_prob if drop_path_prob > 0. else 0.)
        self.norm2 = norm_layer(epsilon=1e-5, name=f'{self.prefix}/norm2')
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       drop=drop, prefix=self.prefix)

    def build(self, input_shape):
        if self.shift_size > 0:
            L = self.input_resolution
            img_mask = np.zeros([1, L, 1])
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None),
                        slice(0, self.shift_size),
                        slice(self.shift_size, self.window_size))
            cnt = 0
            for h in h_slices:
                img_mask[:, h, :] = cnt
                cnt += 1

            img_mask = tf.convert_to_tensor(img_mask)
            mask_windows = window_partition(img_mask, self.window_size) # b * L//win, window_size, 1
            mask_windows = tf.reshape(mask_windows, shape=[-1, self.window_size ]) # nW, window_size
            # [nw, 1, window_size] - [nw, window_size, 1]
            # 通过广播的形式来相减
            # 目的是计算Attention的时候，让具有相同index QK进行计算，而忽略不同index QK计算结果。
            # 使下标相同的地方值为0,然后对下标不同的地方(值不为0的地方)进行
            # 掩码-100,然后将掩码与QK的结果相加,这样不是相同下标的地方值为一个大负数,
            # 然后通过softmax隐去它的影响
            attn_mask = tf.expand_dims(
                mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2) # b* L// win, window_size, window_size
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(
                initial_value=attn_mask, trainable=False, name=f'{self.prefix}/attn_mask')
        else:
            self.attn_mask = None

        self.built = True

    def call(self, x):
        # H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        # x = tf.reshape(x, shape=[-1, H, W, C])

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=-self.shift_size, axis=1)
        else:
            shifted_x = x

        # partition windows
        # x_windows = window_partition(shifted_x, self.window_size) # nW*B, window_size, C
        # x_windows = tf.reshape(
        #     x_windows, shape=[-1, self.window_size * self.window_size, C])

        # W-MSA/SW-MSA
        attn_windows = self.attn(shifted_x, mask=self.attn_mask) # nW*B, window_size, C

        # merge windows
        # attn_windows = tf.reshape(
        #     attn_windows, shape=[-1, self.window_size, self.window_size, C])
        # shifted_x = window_reverse(attn_windows, self.window_size, L, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(attn_windows, shift=self.shift_size, axis=1)
        else:
            x = attn_windows
        # x = tf.reshape(x, shape=[-1, H * W, C])

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, input_resolution, dim, norm_layer=LayerNormalization, prefix=''):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = Dense(2* dim, use_bias=False,
                               name=f'{prefix}/downsample/reduction')
        self.norm = norm_layer(epsilon=1e-5, name=f'{prefix}/downsample/norm')

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


class BasicLayer(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path_prob=0., norm_layer=LayerNormalization, downsample=None, use_checkpoint=False, prefix=''):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = tf.keras.Sequential([SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                                                num_heads=num_heads, window_size=window_size,
                                                                shift_size=0 if (
                                                                        i % 2 == 0) else window_size // 2,
                                                                shuffle=False if ( i % 2 ==0) else True,
                                                                mlp_ratio=mlp_ratio,
                                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                                drop=drop, attn_drop=attn_drop,
                                                                drop_path_prob=drop_path_prob[i] if isinstance(
                                                                    drop_path_prob, list) else drop_path_prob,
                                                                norm_layer=norm_layer,
                                                                prefix=f'{prefix}/blocks{i}') for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer, prefix=prefix)
        else:
            self.downsample = None

    def call(self, x):
        x = self.blocks(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(tf.keras.layers.Layer):
    def __init__(self, img_size=32768, patch_size=16, in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__(name='patch_embed')
        patches_resolution = img_size // patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = Conv1D(embed_dim, kernel_size=patch_size,
                           strides=patch_size, name='proj', padding='same')
        if norm_layer is not None:
            self.norm = norm_layer(epsilon=1e-5, name='norm')
        else:
            self.norm = None

    def call(self, x):
        B, L, C = x.get_shape().as_list()
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        # x = tf.reshape(
        #     x, shape=[-1, (H // self.patch_size[0]) * (W // self.patch_size[0]), self.embed_dim])
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformerModel(tf.keras.Model):
    def __init__(self, model_name='swin_tiny_patch4_2048_1D', include_top=False,
                 img_size=32768, patch_size=16, in_chans=1, num_classes=2,
                 embed_dim=12, depths=[2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=16, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=LayerNormalization, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__(name=model_name)

        self.include_top = include_top

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute postion embedding
        if self.ape:
            self.absolute_pos_embed = self.add_weight('absolute_pos_embed',
                                                      shape=(
                                                          1, num_patches, embed_dim),
                                                      initializer=tf.initializers.Zeros())

        self.pos_drop = Dropout(drop_rate)

        # stochastic depth
        dpr = [x for x in np.linspace(0., drop_path_rate, sum(depths))]

        # build layers
        self.basic_layers = tf.keras.Sequential([BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                                            input_resolution=(patches_resolution // (4 ** i_layer)),
                                                            depth=depths[i_layer],
                                                            num_heads=num_heads[i_layer],
                                                            window_size=window_size,
                                                            mlp_ratio=self.mlp_ratio,
                                                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                            drop=drop_rate, attn_drop=attn_drop_rate,
                                                            drop_path_prob=dpr[sum(depths[:i_layer]):sum(
                                                                depths[:i_layer + 1])],
                                                            norm_layer=norm_layer,
                                                            downsample=PatchMerging if (
                                                                    i_layer < self.num_layers - 1) else None,
                                                            use_checkpoint=use_checkpoint,
                                                            prefix=f'layers{i_layer}') for i_layer in
                                                 range(self.num_layers)])

        self.norm = norm_layer(epsilon=1e-5, name='norm')
        self.avgpool = GlobalAveragePooling1D()
        self.head = Dense(num_classes, name='head', activation='softmax')

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x = self.basic_layers(x)

        x = self.norm(x)
        x = self.avgpool(x)
        return x

    def call(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x



