import numpy as np
from .utils import im2col, col2im

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError

class Conv2D(Layer):
    def __init__(self, input_shape, num_filters, kernel_size, padding=0, stride=1):
        super().__init__()
        self.input_shape = input_shape # (channels, height, width)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        # Initialize weights and biases
        self.weights = np.random.randn(num_filters, input_shape[0], kernel_size, kernel_size)
        self.biases = np.random.randn(num_filters, 1)

    def forward(self, input):
        self.input = input
        N, C, H, W = input.shape
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        col = im2col(input, self.kernel_size, self.kernel_size, self.stride, self.padding)
        col_W = self.weights.reshape(self.num_filters, -1).T

        out = np.dot(col, col_W) + self.biases.T
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.col = col
        self.output = out
        return self.output

    def backward(self, output_gradient, learning_rate):
        N, C, H, W = self.input.shape
        FN, C, FH, FW = self.weights.shape
        out_h = (H + 2 * self.padding - FH) // self.stride + 1
        out_w = (W + 2 * self.padding - FW) // self.stride + 1

        # Reshape output_gradient to match the shape after dot product in forward
        dout = output_gradient.transpose(0, 2, 3, 1).reshape(-1, FN)

        # Calculate gradients for biases
        self.biases_gradient = np.sum(dout, axis=0, keepdims=True).T

        # Calculate gradients for weights
        self.weights_gradient = np.dot(self.col.T, dout)
        self.weights_gradient = self.weights_gradient.transpose(1, 0).reshape(FN, C, FH, FW)

        # Calculate gradient for input
        dcol = np.dot(dout, self.weights.reshape(FN, -1))
        input_gradient = col2im(dcol, self.input.shape, FH, FW, self.stride, self.padding)

        # Update weights and biases
        self.weights -= learning_rate * self.weights_gradient
        self.biases -= learning_rate * self.biases_gradient

        return input_gradient

class MaxPooling2D(Layer):
    def __init__(self, pool_size, stride=None):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size

    def forward(self, input):
        self.input = input
        batch_size, channels, in_height, in_width = input.shape
        out_height = (in_height - self.pool_size) // self.stride + 1
        out_width = (in_width - self.pool_size) // self.stride + 1

        self.output = np.zeros((batch_size, channels, out_height, out_width))
        self.mask = np.zeros_like(input) # To store indices of max values for backward pass

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        window = input[b, c, h_start:h_end, w_start:w_end]
                        self.output[b, c, i, j] = np.max(window)
                        # Store mask for backprop
                        max_idx = np.unravel_index(np.argmax(window, axis=None), window.shape)
                        self.mask[b, c, h_start + max_idx[0], w_start + max_idx[1]] = 1
        return self.output

    def backward(self, output_gradient, learning_rate):
        dx = np.zeros_like(self.input)
        batch_size, channels, out_height, out_width = output_gradient.shape

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        # Distribute gradient to the max element
                        window_mask = self.mask[b, c, h_start:h_end, w_start:w_end]
                        dx[b, c, h_start:h_end, w_start:w_end] += window_mask * output_gradient[b, c, i, j]
        return dx

class AvgPooling2D(Layer):
    def __init__(self, pool_size, stride=None):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size

    def forward(self, input):
        self.input = input
        batch_size, channels, in_height, in_width = input.shape
        out_height = (in_height - self.pool_size) // self.stride + 1
        out_width = (in_width - self.pool_size) // self.stride + 1

        self.output = np.zeros((batch_size, channels, out_height, out_width))

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        window = input[b, c, h_start:h_end, w_start:w_end]
                        self.output[b, c, i, j] = np.mean(window)
        return self.output

    def backward(self, output_gradient, learning_rate):
        dx = np.zeros_like(self.input)
        batch_size, channels, out_height, out_width = output_gradient.shape
        pool_area = self.pool_size * self.pool_size

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        # Distribute gradient evenly
                        dx[b, c, h_start:h_end, w_start:w_end] += output_gradient[b, c, i, j] / pool_area
        return dx

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.original_shape = None

    def forward(self, input):
        self.input = input
        self.original_shape = input.shape
        return input.reshape(input.shape[0], -1)

    def backward(self, output_gradient, learning_rate):
        return output_gradient.reshape(self.original_shape)

class FullyConnected(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.biases

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient)
        biases_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        input_gradient = np.dot(output_gradient, self.weights.T)

        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient
        return input_gradient

class Dropout(Layer):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate
        self.mask = None

    def forward(self, input):
        self.input = input
        if self.rate > 0:
            self.mask = (np.random.rand(*input.shape) > self.rate) / (1 - self.rate)
            return input * self.mask
        return input

    def backward(self, output_gradient, learning_rate):
        if self.rate > 0:
            return output_gradient * self.mask
        return output_gradient

class BatchNorm2D(Layer):
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        super().__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = np.ones((1, num_features, 1, 1))
        self.beta = np.zeros((1, num_features, 1, 1))
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))

    def forward(self, input, training=True):
        self.input = input
        if training:
            mean = np.mean(input, axis=(0, 2, 3), keepdims=True)
            var = np.var(input, axis=(0, 2, 3), keepdims=True)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            self.x_norm = (input - mean) / np.sqrt(var + self.epsilon)
            self.output = self.gamma * self.x_norm + self.beta
        else:
            self.x_norm = (input - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            self.output = self.gamma * self.x_norm + self.beta
        return self.output

    def backward(self, output_gradient, learning_rate):
        N, C, H, W = self.input.shape

        # Gradients for beta
        dbeta = np.sum(output_gradient, axis=(0, 2, 3), keepdims=True)

        # Gradients for gamma
        dgamma = np.sum(output_gradient * self.x_norm, axis=(0, 2, 3), keepdims=True)

        # Gradients for normalized input (x_norm)
        dx_norm = output_gradient * self.gamma

        # Gradients for variance and mean
        var = self.running_var if not hasattr(self, 'mean') else np.var(self.input, axis=(0, 2, 3), keepdims=True)
        mean = self.running_mean if not hasattr(self, 'mean') else np.mean(self.input, axis=(0, 2, 3), keepdims=True)

        dvar = np.sum(dx_norm * (self.input - mean) * -0.5 * np.power(var + self.epsilon, -1.5), axis=(0, 2, 3), keepdims=True)
        dmean = np.sum(dx_norm * -1 / np.sqrt(var + self.epsilon), axis=(0, 2, 3), keepdims=True) +\
                dvar * np.mean(-2 * (self.input - mean), axis=(0, 2, 3), keepdims=True)

        # Gradients for input (x)
        dx = dx_norm / np.sqrt(var + self.epsilon) +\
             dvar * 2 * (self.input - mean) / N / C / H / W +\
             dmean / N / C / H / W

        # Update gamma and beta
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta

        return dx

class ResidualBlock(Layer):
    def __init__(self, input_channels, num_filters):
        super().__init__()
        self.conv1 = Conv2D(input_shape=(input_channels, 1, 1), num_filters=num_filters // 2, kernel_size=1)
        self.bn1 = BatchNorm2D(num_features=num_filters // 2)
        self.relu1 = ReLU()
        self.conv2 = Conv2D(input_shape=(num_filters // 2, 1, 1), num_filters=num_filters, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2D(num_features=num_filters)
        self.relu2 = ReLU()

    def forward(self, input):
        self.input = input
        residual = input

        out = self.conv1.forward(input)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)

        out = self.conv2.forward(out)
        out = self.bn2.forward(out)
        out = self.relu2.forward(out)

        # Assuming input and output shapes are compatible for addition
        # This needs careful handling for spatial dimensions if stride > 1 in conv layers
        # For Darknet-53, residual connections are usually after a 1x1 and 3x3 conv sequence
        # where spatial dimensions are preserved.
        self.output = out + residual
        return self.output

    def backward(self, output_gradient, learning_rate):
        # Gradient for the addition operation is simply the output_gradient passed to both paths
        d_residual = output_gradient

        # Backward through the main path
        d_out = self.relu2.backward(output_gradient)
        d_out = self.bn2.backward(d_out, learning_rate) # Pass learning_rate for BatchNorm
        d_out = self.conv2.backward(d_out, learning_rate) # Pass learning_rate for Conv2D

        d_out = self.relu1.backward(d_out)
        d_out = self.bn1.backward(d_out, learning_rate) # Pass learning_rate for BatchNorm
        d_out = self.conv1.backward(d_out, learning_rate) # Pass learning_rate for Conv2D

        # Sum gradients from both paths
        input_gradient = d_out + d_residual
        return input_gradient

class Route(Layer):
    def __init__(self, layers_to_route):
        super().__init__()
        self.layers_to_route = layers_to_route # List of indices or layer objects to concatenate
        self.inputs = []

    def forward(self, inputs):
        # Inputs will be a list of feature maps from previous layers
        # For simplicity, assuming concatenation along the channel dimension
        self.inputs = inputs
        self.output = np.concatenate(inputs, axis=1)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # Distribute the gradient back to the original input paths
        # This requires knowing the original shapes of the concatenated inputs
        gradients = []
        current_idx = 0
        for input_tensor in self.inputs:
            channels = input_tensor.shape[1]
            gradients.append(output_gradient[:, current_idx:current_idx + channels, :, :])
            current_idx += channels
        return gradients # Return a list of gradients

class YoloOutput(Layer):
    def __init__(self, num_classes, anchors, num_bounding_boxes, input_shape):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = np.array(anchors).reshape(-1, 2) # Reshape anchors to (num_anchors, 2)
        self.num_bounding_boxes = num_bounding_boxes
        self.input_shape = input_shape # (batch, channels, H, W)

    def forward(self, input_data):
        # input_data shape: (batch_size, num_anchors * (5 + num_classes), grid_h, grid_w)
        N, C, H, W = input_data.shape

        # Reshape to (batch_size, num_anchors, (5 + num_classes), grid_h, grid_w)
        # Then permute to (batch_size, grid_h, grid_w, num_anchors, (5 + num_classes))
        prediction = input_data.reshape(N, self.num_bounding_boxes, 5 + self.num_classes, H, W)
        prediction = prediction.transpose(0, 3, 4, 1, 2) # (N, H, W, num_anchors, 5 + num_classes)

        # Extract components
        # tx, ty, tw, th, objectness, class_scores
        x_center = prediction[..., 0:1] # tx
        y_center = prediction[..., 1:2] # ty
        width = prediction[..., 2:3]    # tw
        height = prediction[..., 3:4]   # th
        objectness = prediction[..., 4:5] # objectness score
        class_scores = prediction[..., 5:] # class scores

        # Apply sigmoid to x_center, y_center, and objectness
        x_center = 1 / (1 + np.exp(-x_center))
        y_center = 1 / (1 + np.exp(-y_center))
        objectness = 1 / (1 + np.exp(-objectness))

        # Apply exponential to width and height, and multiply by anchors
        # Grid coordinates (cx, cy) need to be added here. This requires knowing the grid size.
        # For now, let's assume grid_x and grid_y are available or calculated externally.
        # This is a placeholder for the actual calculation.
        # For a full implementation, grid_x and grid_y would be generated based on H, W
        grid_x = np.arange(W).reshape(1, 1, W, 1, 1)
        grid_y = np.arange(H).reshape(1, H, 1, 1, 1)

        box_x = x_center + grid_x
        box_y = y_center + grid_y
        box_w = np.exp(width) * self.anchors[:, 0].reshape(1, 1, 1, self.num_bounding_boxes, 1)
        box_h = np.exp(height) * self.anchors[:, 1].reshape(1, 1, 1, self.num_bounding_boxes, 1)

        # Apply softmax to class scores
        exp_class_scores = np.exp(class_scores - np.max(class_scores, axis=-1, keepdims=True))
        class_probabilities = exp_class_scores / np.sum(exp_class_scores, axis=-1, keepdims=True)

        # Combine outputs
        self.output = np.concatenate([box_x, box_y, box_w, box_h, objectness, class_probabilities], axis=-1)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # Placeholder for backward pass, will be handled during YOLO loss calculation
        pass

    def decode_output(self, output, anchors, img_size, num_classes, stride):
        # output: (batch_size, grid_h, grid_w, num_anchors * (5 + num_classes))
        # anchors: (num_anchors, 2) - anchor widths and heights
        # img_size: (img_h, img_w)
        # num_classes: number of classes
        # stride: stride of the feature map (e.g., 32, 16, 8)

        batch_size, grid_h, grid_w, _ = output.shape
        num_anchors = len(anchors)

        # Reshape output to (batch_size, grid_h, grid_w, num_anchors, 5 + num_classes)
        prediction = output.reshape(batch_size, grid_h, grid_w, num_anchors, 5 + num_classes)

        # Get x, y, w, h, objectness, class_scores
        x = sigmoid(prediction[..., 0])
        y = sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        obj_score = sigmoid(prediction[..., 4])
        class_scores = sigmoid(prediction[..., 5:])

        # Create grid for x, y offsets
        grid_x = np.arange(grid_w).reshape(1, 1, grid_w, 1, 1)
        grid_y = np.arange(grid_h).reshape(1, grid_h, 1, 1, 1)

        # Adjust x, y coordinates based on grid and stride
        # (x + grid_x) * stride
        # (y + grid_y) * stride
        pred_boxes_x = (x + grid_x) * stride
        pred_boxes_y = (y + grid_y) * stride

        # Adjust w, h based on anchors and stride
        # exp(w) * anchor_w * stride
        # exp(h) * anchor_h * stride
        anchors_w = anchors[:, 0].reshape(1, 1, 1, num_anchors, 1)
        anchors_h = anchors[:, 1].reshape(1, 1, 1, num_anchors, 1)

        pred_boxes_w = np.exp(w) * anchors_w
        pred_boxes_h = np.exp(h) * anchors_h

        # Convert to (x1, y1, x2, y2) format for easier NMS later
        # x1 = center_x - w/2
        # y1 = center_y - h/2
        # x2 = center_x + w/2
        # y2 = center_y + h/2
        pred_boxes_x1 = pred_boxes_x - pred_boxes_w / 2
        pred_boxes_y1 = pred_boxes_y - pred_boxes_h / 2
        pred_boxes_x2 = pred_boxes_x + pred_boxes_w / 2
        pred_boxes_y2 = pred_boxes_y + pred_boxes_h / 2

        # Normalize coordinates to image size (0-1 range)
        img_w, img_h = img_size
        pred_boxes_x1 /= img_w
        pred_boxes_y1 /= img_h
        pred_boxes_x2 /= img_w
        pred_boxes_y2 /= img_h

        # Combine all predictions
        # (batch_size, grid_h, grid_w, num_anchors, 4 + 1 + num_classes)
        # 4 for bbox (x1,y1,x2,y2), 1 for objectness, num_classes for class scores
        decoded_predictions = np.concatenate([
            pred_boxes_x1[..., np.newaxis],
            pred_boxes_y1[..., np.newaxis],
            pred_boxes_x2[..., np.newaxis],
            pred_boxes_y2[..., np.newaxis],
            obj_score[..., np.newaxis],
            class_scores
        ], axis=-1)

        return decoded_predictions