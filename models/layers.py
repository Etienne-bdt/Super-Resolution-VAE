import math

import torch
import torch.nn as nn


def calculate_padding(in_size, out_size, kernel_size, stride=1, dilation=1):
    """
    Calculate the padding needed for a convolutional layer given the kernel size and dilation.
    """
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    padding = max(0, (in_size - out_size * stride + effective_kernel_size - 1) // 2)
    return padding


def calculate_output_size(in_size, kernel_size, stride=1, padding=0, dilation=1):
    """
    Calculate the output size of a convolutional layer given the input size, kernel size, stride, padding, and dilation.
    """
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    out_size = (in_size + 2 * padding - effective_kernel_size) // stride + 1
    return out_size


class downsample_sequence(nn.Module):
    def __init__(self, in_shape, compression_ratio, num_steps=None):
        """
        Args:
            in_shape: (C, H, W) tuple for input
            out_flattened_size: int, desired flattened output size (must be divisible by out_channels and form a square shape)
            out_channels: output channels (optional, will be auto-calculated if not given)
            num_steps: number of downsampling steps (optional, will be inferred if not given)
        """
        super(downsample_sequence, self).__init__()
        out_flattened_size = int(math.prod(in_shape) / compression_ratio)
        if out_flattened_size % 2 != 0:
            out_flattened_size += 1
        assert out_flattened_size is not None, "Must specify out_flattened_size"
        # Auto-calculate out_channels and H=W if not provided

        self.out_channels = out_flattened_size
        self.target_h = 1
        self.target_w = 1
        self.layers = nn.ModuleList()
        c, h, w = in_shape[0], in_shape[1], in_shape[2]
        steps = num_steps or 0
        # If num_steps not given, compute minimal steps to reach target spatial size
        if num_steps is None:
            tmp_h, tmp_w = h, w
            while tmp_h > self.target_h and tmp_w > self.target_w:
                tmp_h = (tmp_h + 1) // 2
                tmp_w = (tmp_w + 1) // 2
                steps += 1
        else:
            steps = num_steps

        if steps > 0:
            max_stride_steps = 0
            tmp_h, tmp_w = h, w
            for _ in range(steps):
                if tmp_h > self.target_h and tmp_w > self.target_w:
                    tmp_h = (tmp_h + 1) // 2
                    tmp_w = (tmp_w + 1) // 2
                    max_stride_steps += 1
                else:
                    break
            # Use stride=2 for max_stride_steps, then stride=1 for the rest
            stride_plan = [2] * max_stride_steps + [1] * (steps - max_stride_steps)
        else:
            stride_plan = []

        # Plan progressive channel increase using powers of four
        if steps > 1:
            ch_progression = [
                min(self.out_channels, in_shape[0] * (4**i)) for i in range(steps)
            ]
        else:
            ch_progression = [self.out_channels]
        for i in range(steps):
            is_last = i == steps - 1
            stride = stride_plan[i] if i < len(stride_plan) else 1
            kernel_size = 5 if stride == 2 else 3
            next_ch = ch_progression[i]
            out_ch = next_ch
            # For last layer, match target_h/target_w
            next_h = self.target_h if is_last else (h + stride - 1) // stride
            padding = calculate_padding(h, next_h, kernel_size, stride)
            self.layers.append(nn.Conv2d(c, c, 3, stride=1, padding=1))
            self.layers.append(
                nn.Conv2d(c, out_ch, kernel_size, stride=stride, padding=padding)
            )
            self.layers.append(nn.BatchNorm2d(out_ch))
            # If not last layer, add ReLU activation
            if not is_last:
                self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(self_attention(out_ch, num_heads=2))
            c = out_ch
            h = calculate_output_size(h, kernel_size, stride, padding)
            w = calculate_output_size(w, kernel_size, stride, padding)
        self.final_shape = (c, h, w)

        self.flatten = nn.Flatten()
        assert c * h * w == out_flattened_size, (
            f"Final output shape {c}x{h}x{w} does not match requested flattened size {out_flattened_size}"
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.flatten(x)
        return x


class upsample_sequence(nn.Module):
    def __init__(self, in_flattened_size, out_shape, num_steps=None):
        """
        Args:
            in_flattened_size: int, flattened input size (must be divisible by in_channels and form a square shape)
            out_shape: (C, H, W) tuple for output
            in_channels: number of channels to unflatten to (should match last out_channels of downsample)
            num_steps: number of upsampling steps (optional, will be inferred if not given)
        """
        super(upsample_sequence, self).__init__()
        assert in_flattened_size is not None, "Must specify in_flattened_size"
        assert out_shape is not None, "Must specify out_shape (C, H, W)"
        in_channels = in_flattened_size
        out_channels = out_shape[0]  # Always infer from out_shape
        self.sigmoid = nn.Sigmoid()
        target_h, target_w = out_shape[1], out_shape[2]
        self.out_channels = out_channels
        self.target_h = target_h
        self.target_w = target_w
        self.layers = nn.ModuleList()
        # Infer in_h, in_w from in_flattened_size and in_channels

        total_spatial = 1
        in_h = in_w = int(total_spatial**0.5)
        assert in_h * in_w * in_channels == in_flattened_size, (
            "in_flattened_size must be divisible by in_channels and form a square shape"
        )
        self.in_channels = in_channels
        self.in_h = in_h
        self.in_w = in_w
        c, h, w = in_channels, in_h, in_w
        # If num_steps not given, compute minimal steps to reach target spatial size
        steps = 0
        tmp_h, tmp_w = h, w
        while tmp_h < target_h and tmp_w < target_w:
            tmp_h = tmp_h * 2
            tmp_w = tmp_w * 2
            steps += 1
        if num_steps is not None and steps < num_steps:
            steps = num_steps

        max_stride_steps = 0
        tmp_h, tmp_w = h, w
        for _ in range(steps):
            if tmp_h < target_h and tmp_w < target_w:
                tmp_h = tmp_h * 2
                tmp_w = tmp_w * 2
                max_stride_steps += 1
            else:
                break
        stride_plan = [2] * max_stride_steps + [1] * (steps - max_stride_steps)
        if steps > 1:
            ch_progression = [max(out_channels, c // (4**i)) for i in range(steps)]
            ch_progression[-1] = out_channels  # Ensure last is exactly out_channels
        else:
            ch_progression = [out_channels]
        self.unflatten = nn.Unflatten(1, (in_channels, in_h, in_w))
        # Improved output shape calculation for ConvTranspose2d (no padding)
        # Use kernel_size=4, stride=2, padding=1 for upsampling (standard for exact doubling)
        for i in range(steps):
            is_last = i == steps - 1
            stride = stride_plan[i] if i < len(stride_plan) else 1
            if stride == 2:
                kernel_size = 4
                padding = 1
            else:
                kernel_size = 3
                padding = 1
            next_ch = ch_progression[i]
            out_ch = next_ch
            # output_padding must be int, not tuple
            output_padding = 0
            self.layers.append(nn.Conv2d(c, c, 3, stride=1, padding=1))
            self.layers.append(
                nn.ConvTranspose2d(
                    c,
                    out_ch,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            )
            if not is_last:
                self.layers.append(nn.ReLU(inplace=True))
            # Update h, w using the ConvTranspose2d formula
            h = (h - 1) * stride - 2 * padding + kernel_size + output_padding
            w = (w - 1) * stride - 2 * padding + kernel_size + output_padding
            c = out_ch
        self.final_shape = (c, h, w)
        if not (c == out_channels and h == target_h and w == target_w):
            raise RuntimeError(
                f"Upsample sequence produced invalid shape {c}x{h}x{w}, expected {out_channels}x{target_h}x{target_w}. Check your configuration."
            )

    def forward(self, x):
        x = self.unflatten(x)
        for layer in self.layers:
            x = layer(x)
        x = self.sigmoid(x)
        return x


class down_block(nn.Module):
    def __init__(self, in_channels, out_channels, with_relu=True, with_bn=True):
        """
        Downsampling block with convolution, batch normalization, and ReLU activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolution kernel (default: 3).
            stride (int): Stride for the convolution (default: 2).
        """
        super(down_block, self).__init__()
        self.with_relu = with_relu
        self.with_bn = with_bn
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.downsample = nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.bn_in = nn.BatchNorm2d(in_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass through the downsampling block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, new_height, new_width).
        """
        x = self.conv(x)
        x = self.relu(x) if self.with_relu else x
        x = self.bn_in(x) if self.with_bn else x
        x = self.downsample(x)
        if self.with_relu:
            x = self.relu(x)
        if self.with_bn:
            x = self.bn(x)
        return x


class up_block(nn.Module):
    def __init__(self, in_channels, out_channels, with_relu=True, with_bn=True):
        """
        Upsampling block with transposed convolution, batch normalization, and ReLU activation.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the transposed convolution kernel (default: 3).
            stride (int): Stride for the transposed convolution (default: 2).
        """
        super(up_block, self).__init__()
        self.with_relu = with_relu
        self.with_bn = with_bn
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.bn_in = nn.BatchNorm2d(in_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass through the upsampling block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, new_height, new_width).
        """
        x = self.conv(x)
        x = self.relu(x) if self.with_relu else x
        x = self.bn_in(x) if self.with_bn else x
        x = self.upsample(x)
        if self.with_relu:
            x = self.relu(x)
        if self.with_bn:
            x = self.bn(x)
        return x


class self_attention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        """
        Convolutional self-attention layer.

        Args:
            in_channels (int): Number of input channels.
            num_heads (int): Number of attention heads (default: 8).
        """
        super(self_attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert in_channels % num_heads == 0, (
            "in_channels must be divisible by num_heads"
        )

        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        # Compute query, key, value
        query = self.query_conv(x).view(
            batch_size, self.num_heads, self.head_dim, height * width
        )
        key = self.key_conv(x).view(
            batch_size, self.num_heads, self.head_dim, height * width
        )
        value = self.value_conv(x).view(
            batch_size, self.num_heads, self.head_dim, height * width
        )

        # Transpose to get (batch_size, num_heads, height * width, head_dim)
        query = query.permute(0, 1, 3, 2)
        key = key.permute(0, 1, 3, 2)
        value = value.permute(0, 1, 3, 2)
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (
            self.head_dim**0.5
        )
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply attention weights to value
        out = torch.matmul(attention_weights, value)
        out = (
            out.permute(0, 1, 3, 2)
            .contiguous()
            .view(batch_size, channels, height, width)
        )

        # Final convolution to combine heads
        out = self.out_conv(out)
        return out + x


class residual(nn.Module):
    def __init__(self, module):
        """
        Residual connection wrapper for a module.

        Args:
            module (nn.Module): The module to wrap with a residual connection.
        """
        super(residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


if __name__ == "__main__":
    # Example usage
    in_shape = (4, 32, 32)  # C, H, W
    compression_ratio = 1.5
    model = downsample_sequence(
        in_shape,
        compression_ratio / 2,
    )
    latent_size = int(math.prod(in_shape) // compression_ratio)

    print(model)

    # Test with a random input tensor
    x = torch.randn(1, *in_shape)  # Batch size of 1
    output = model(x)
    output, o = torch.chunk(output, 2, dim=1)
    print("Output shape:", output.shape)

    # Example usage for upsample_sequence
    upmodel = upsample_sequence(
        in_flattened_size=latent_size,
        out_shape=(4, 32, 32),
    )
    print(upmodel)

    upoutput = upmodel(output)
    print("Upsample output shape:", upoutput.shape)

    attention = self_attention(in_channels=256, num_heads=8)
    x_att = torch.randn(1, 256, 16, 16)
    att_output = attention(x_att)
    print("Attention output shape:", att_output.shape)
