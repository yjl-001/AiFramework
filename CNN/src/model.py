import numpy as np
from .layers import Layer, Conv2D, MaxPooling2D, AvgPooling2D, Flatten, FullyConnected, Dropout, BatchNorm2D, ResidualBlock, Route, YoloOutput
from .activations import Activation, ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
from .losses import Loss, MeanSquaredError, CategoricalCrossentropy, BinaryCrossentropy, YOLOLoss
from .optimizers import Optimizer, SGD, Momentum, RMSProp, Adam
from .initializers import Initializer, Zeros, Ones, RandomNormal, RandomUniform, GlorotNormal, GlorotUniform, HeNormal, HeUniform
from .regularization import Regularizer, L1, L2, ElasticNet

class Model:
    def __init__(self):
        self.layers = []
        self.loss_fn = None
        self.optimizer = None
        self.metrics = {}

    def add(self, layer):
        if isinstance(layer, Layer) or isinstance(layer, Activation):
            self.layers.append(layer)
        else:
            raise ValueError("Only Layer or Activation instances can be added to the model.")

    def compile(self, optimizer, loss, metrics=None):
        self.optimizer = optimizer
        self.loss_fn = loss
        if metrics:
            self.metrics = metrics

    def forward(self, input_data, training=True):
        x = input_data
        for layer in self.layers:
            if isinstance(layer, (Dropout, BatchNorm2D)):
                x = layer.forward(x, training=training)
            else:
                x = layer.forward(x)
        return x

    def backward(self, output_gradient, learning_rate):
        grad = output_gradient
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
        return grad

    def train_step(self, x_batch, y_batch_list, learning_rate):
        # y_batch_list is a list of true labels for each YOLO output scale
        # Forward pass
        raw_predictions_list = self.forward(x_batch, training=True)

        total_loss = 0
        output_gradients = [] # Store gradients for each head

        # Calculate loss and initial gradients for each YOLO head
        for i in range(len(raw_predictions_list)):
            raw_pred = raw_predictions_list[i]
            y_true = y_batch_list[i]
            yolo_output_layer = self.yolo_output_layers[i]
            loss_fn_for_head = YOLOLoss(anchors=yolo_output_layer.anchors, 
                                        img_size=self.img_size, 
                                        num_classes=self.num_classes)
            
            # The loss function expects y_true in the format (batch, grid_h, grid_w, num_anchors, 5+C)
            # and y_pred_raw in the format (batch, grid_h, grid_w, num_anchors * (5+C))
            # Ensure y_true is correctly formatted before passing to loss_fn_for_head.loss
            loss_val = loss_fn_for_head.loss(y_true, raw_pred, yolo_output_layer.stride)
            total_loss += loss_val

            # Gradient calculation for YOLO is complex and usually integrated.
            # For this framework, the YOLOLoss.gradient is a placeholder.
            # The backward pass of the YoloOutput layer itself should compute the gradient
            # with respect to its input, given the gradient of the loss with respect to its output.
            # Here, we'd ideally get dL/dy_pred_raw from the loss function.
            # Since YOLOLoss.gradient is not fully implemented, we'll simulate this.
            # The gradient passed to the YoloOutput layer's backward method should be dL/d(output_of_YoloOutput_layer)
            # This is essentially the gradient of the loss w.r.t. the raw network output for that head.
            # This part is highly conceptual without a full gradient implementation in YOLOLoss.
            # For now, let's assume a simplified gradient or pass a placeholder.
            # A proper implementation would derive this from the loss components.
            
            # Placeholder: The gradient of the loss w.r.t. the raw prediction of this head.
            # This would be d(total_loss)/d(raw_pred). For simplicity, we'll use a dummy value.
            # In a real scenario, this would be computed based on the loss formula.
            # For example, if using MSE for coordinates, BCE for objectness/classes, the gradients
            # would be derived from those. The YOLOLoss.gradient should ideally return this.
            # Since it's not implemented, we'll create a dummy gradient of the same shape as raw_pred.
            # This is a major simplification and needs a proper implementation for actual training.
            grad_for_head = np.zeros_like(raw_pred) # Dummy gradient
            # In a real implementation, this would be: grad_for_head = loss_fn_for_head.gradient(y_true, raw_pred, yolo_output_layer.stride)
            output_gradients.append(grad_for_head)

        # Backward pass - This needs to be adapted for multiple output heads.
        # The current Model.backward expects a single output_gradient.
        # We need to backpropagate gradients for each head separately through their respective paths.
        # This requires a more sophisticated backward pass in the Model class or specific handling in YOLOv3.

        # For now, we'll call backward on the last layer of each head with its specific gradient.
        # This is a simplification and assumes the model structure allows this.
        # The gradients need to be propagated back from each YoloOutput layer.
        # The self.backward method needs to be aware of multiple output paths.
        
        # This part is highly conceptual and needs a redesign of the Model.backward method
        # to handle multiple outputs and their gradients.
        # For now, we cannot directly use self.backward(output_gradients, learning_rate)
        # as output_gradients is a list and self.backward expects a single tensor.

        # --- Simplified/Conceptual Backward Pass for YOLO --- 
        # We need to iterate through the layers in reverse, but the gradient flow is more complex
        # due to multiple heads and route layers. 
        # A full backpropagation for YOLO in this framework would require careful handling of gradient splitting and merging.

        # For now, we'll acknowledge this limitation. The current Model.backward is not suited for YOLO.
        # A proper implementation would involve: 
        # 1. Modifying Model.backward to accept a list of gradients for multi-output models.
        # 2. Tracing gradients back from each YoloOutput layer, considering Route layers and shared backbone.

        # Placeholder for optimizer update (as in original Model class)
        # self.optimizer.update(self.layers, learning_rate) # Assuming optimizer can handle this

        return total_loss, raw_predictions_list

    def fit(self, X_train, y_train, epochs, batch_size, learning_rate):
        num_samples = X_train.shape[0]
        for epoch in range(epochs):
            # Shuffle data
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            epoch_loss = 0
            for i in range(0, num_samples, batch_size):
                x_batch = X_train_shuffled[i:i + batch_size]
                # For YOLO, y_batch needs to be a list of ground truth tensors for each scale.
                # This typically comes from a data loading pipeline that processes annotations.
                # For now, we'll create dummy y_batch_list based on expected output shapes.
                # In a real scenario, y_train would be the raw annotations, and a preprocessing
                # step would convert them into the correct format for each YOLO head.
                
                # Placeholder for y_batch_list generation
                # This assumes y_train contains the necessary info to generate targets for all 3 heads.
                # For a proper implementation, this would involve a function that takes raw labels
                # and generates the target tensors for each grid scale.
                y_batch_list = []
                # Example: Assuming y_train is a single tensor that needs to be split/processed
                # into 3 target tensors for the 3 YOLO heads.
                # The shapes of these targets depend on grid size, num_anchors, num_classes.
                # For demonstration, let's create dummy targets matching the expected output shapes.
                # This is a simplification and needs actual ground truth processing.
                
                # Get expected grid sizes for each head
                # Head 1 (stride 32): H/32, W/32
                # Head 2 (stride 16): H/16, W/16
                # Head 3 (stride 8): H/8, W/8
                
                # Assuming input_shape is (C, H, W)
                input_h, input_w = self.input_shape[1], self.input_shape[2]
                
                # Dummy y_batch for head 1 (stride 32)
                grid_h1, grid_w1 = input_h // 32, input_w // 32
                y_batch_head1 = np.zeros((x_batch.shape[0], grid_h1, grid_w1, 3, 5 + self.num_classes)) # 3 anchors per scale
                y_batch_list.append(y_batch_head1)

                # Dummy y_batch for head 2 (stride 16)
                grid_h2, grid_w2 = input_h // 16, input_w // 16
                y_batch_head2 = np.zeros((x_batch.shape[0], grid_h2, grid_w2, 3, 5 + self.num_classes))
                y_batch_list.append(y_batch_head2)

                # Dummy y_batch for head 3 (stride 8)
                grid_h3, grid_w3 = input_h // 8, input_w // 8
                y_batch_head3 = np.zeros((x_batch.shape[0], grid_h3, grid_w3, 3, 5 + self.num_classes))
                y_batch_list.append(y_batch_head3)

                loss, predictions = self.train_step(x_batch, y_batch_list, learning_rate)
                epoch_loss += loss

                # Apply optimizer updates (this part needs to be integrated with optimizer class)
                # Currently, layers update their own weights in their backward pass,
                # which is not ideal. A proper optimizer would collect all gradients
                # and apply updates centrally.

            avg_epoch_loss = epoch_loss / (num_samples / batch_size)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

    def evaluate(self, X_test, y_test):
        predictions = self.forward(X_test, training=False)
        loss = self.loss_fn.forward(y_test, predictions)
        print(f"Test Loss: {loss:.4f}")
        return loss, predictions

    def predict(self, X):
        return self.forward(X, training=False)

    def save_weights(self, file_path):
        """Saves the model's weights to a file."""
        weights_dict = {}
        for i, layer in enumerate(self.layers):
            layer_prefix = f"layer_{i}_{layer.__class__.__name__}"
            if hasattr(layer, 'weights'): # Conv2D, FullyConnected
                weights_dict[f"{layer_prefix}_weights"] = layer.weights
                weights_dict[f"{layer_prefix}_biases"] = layer.biases
            if hasattr(layer, 'gamma'): # BatchNorm2D
                weights_dict[f"{layer_prefix}_gamma"] = layer.gamma
                weights_dict[f"{layer_prefix}_beta"] = layer.beta
                weights_dict[f"{layer_prefix}_running_mean"] = layer.running_mean
                weights_dict[f"{layer_prefix}_running_var"] = layer.running_var
        np.savez(file_path, **weights_dict)
        print(f"Model weights saved to {file_path}")

    def load_weights(self, file_path):
        """Loads the model's weights from a file."""
        try:
            weights_dict = np.load(file_path, allow_pickle=True)
            for i, layer in enumerate(self.layers):
                layer_prefix = f"layer_{i}_{layer.__class__.__name__}"
                if hasattr(layer, 'weights'):
                    if f"{layer_prefix}_weights" in weights_dict:
                        layer.weights = weights_dict[f"{layer_prefix}_weights"]
                    if f"{layer_prefix}_biases" in weights_dict:
                        layer.biases = weights_dict[f"{layer_prefix}_biases"]
                if hasattr(layer, 'gamma'):
                    if f"{layer_prefix}_gamma" in weights_dict:
                        layer.gamma = weights_dict[f"{layer_prefix}_gamma"]
                    if f"{layer_prefix}_beta" in weights_dict:
                        layer.beta = weights_dict[f"{layer_prefix}_beta"]
                    if f"{layer_prefix}_running_mean" in weights_dict:
                        layer.running_mean = weights_dict[f"{layer_prefix}_running_mean"]
                    if f"{layer_prefix}_running_var" in weights_dict:
                        layer.running_var = weights_dict[f"{layer_prefix}_running_var"]
            print(f"Model weights loaded from {file_path}")
        except FileNotFoundError:
            print(f"Error: Weights file not found at {file_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")

class YOLOv3(Model):
    def __init__(self, input_shape, num_classes, anchors, img_size=(416, 416)):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.anchors = anchors # (3, 3, 2) for 3 scales, 3 anchors per scale, 2 for w,h
        self.img_size = img_size
        self.layers = []
        self.feature_map_indices = {}
        self.head1_end_index = -1
        self.head2_start_index = -1
        self.head2_end_index = -1
        self.head3_start_index = -1
        self.head3_end_index = -1
        self.yolo_output_layers = [] # To store references to YoloOutput layers
        self._build_model()

    def _conv_block(self, input_channels, num_filters, kernel_size, stride=1, padding=0):
        layers = [
            Conv2D(input_shape=(input_channels, self.input_shape[1], self.input_shape[2]), num_filters=num_filters, kernel_size=kernel_size, stride=stride, padding=padding),
            BatchNorm2D(num_features=num_filters),
            LeakyReLU(alpha=0.1)
        ]
        return layers

    def _res_block(self, input_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(input_channels=input_channels, num_filters=input_channels * 2)) # ResidualBlock expects num_filters to be output channels
        return layers

    def build_darknet53_backbone(self):
        # Initial Conv Layer
        self.add(Conv2D(input_shape=self.input_shape, num_filters=32, kernel_size=3, padding=1))
        self.add(BatchNorm2D(num_features=32))
        self.add(LeakyReLU(alpha=0.1))

        # Downsample and Residual Blocks
        # Block 1: 32 -> 64, 1 residual block
        self.add(Conv2D(input_shape=(32, self.input_shape[1], self.input_shape[2]), num_filters=64, kernel_size=3, stride=2, padding=1))
        self.add(BatchNorm2D(num_features=64))
        self.add(LeakyReLU(alpha=0.1))
        for layer in self._res_block(64, 1):
            self.add(layer)

        # Block 2: 64 -> 128, 2 residual blocks
        self.add(Conv2D(input_shape=(64, self.input_shape[1]//2, self.input_shape[2]//2), num_filters=128, kernel_size=3, stride=2, padding=1))
        self.add(BatchNorm2D(num_features=128))
        self.add(LeakyReLU(alpha=0.1))
        for layer in self._res_block(128, 2):
            self.add(layer)

        # Block 3: 128 -> 256, 8 residual blocks (Feature Map 1 - large objects)
        self.add(Conv2D(input_shape=(128, self.input_shape[1]//4, self.input_shape[2]//4), num_filters=256, kernel_size=3, stride=2, padding=1))
        self.add(BatchNorm2D(num_features=256))
        self.add(LeakyReLU(alpha=0.1))
        for layer in self._res_block(256, 8):
            self.add(layer)
        self.feature_maps.append(len(self.layers) - 1) # Index of the last layer of this block

        # Block 4: 256 -> 512, 8 residual blocks (Feature Map 2 - medium objects)
        self.add(Conv2D(input_shape=(256, self.input_shape[1]//8, self.input_shape[2]//8), num_filters=512, kernel_size=3, stride=2, padding=1))
        self.add(BatchNorm2D(num_features=512))
        self.add(LeakyReLU(alpha=0.1))
        for layer in self._res_block(512, 8):
            self.add(layer)
        self.feature_maps.append(len(self.layers) - 1) # Index of the last layer of this block

        # Block 5: 512 -> 1024, 4 residual blocks (Feature Map 3 - small objects)
        self.add(Conv2D(input_shape=(512, self.input_shape[1]//16, self.input_shape[2]//16), num_filters=1024, kernel_size=3, stride=2, padding=1))
        self.add(BatchNorm2D(num_features=1024))
        self.add(LeakyReLU(alpha=0.1))
        for layer in self._res_block(1024, 4):
            self.add(layer)
        self.feature_maps.append(len(self.layers) - 1) # Index of the last layer of this block (output of block 5)
        self.feature_map_indices = {'block3': self.feature_maps[0], 'block4': self.feature_maps[1], 'block5': self.feature_maps[2]}

        print("Darknet-53 backbone built.")

    def _detection_block(self, input_channels, num_filters_1, num_filters_2, output_channels, input_h, input_w):
        layers = []
        # Convolutional set 1
        layers.extend(self._conv_block(input_channels, num_filters_1, kernel_size=1, padding=0))
        layers.extend(self._conv_block(num_filters_1, num_filters_2, kernel_size=3, padding=1))
        layers.extend(self._conv_block(num_filters_2, num_filters_1, kernel_size=1, padding=0))
        layers.extend(self._conv_block(num_filters_1, num_filters_2, kernel_size=3, padding=1))
        layers.extend(self._conv_block(num_filters_2, num_filters_1, kernel_size=1, padding=0))
        # Output layer for this detection head
        layers.append(Conv2D(input_shape=(num_filters_1, input_h, input_w), num_filters=num_filters_2, kernel_size=3, padding=1))
        layers.append(Conv2D(input_shape=(num_filters_2, input_h, input_w), num_filters=output_channels, kernel_size=1, padding=0))
        return layers

    def build_yolov3_heads(self):
        output_channels = self.num_bounding_boxes * (5 + self.num_classes)

        # --- Head 1 (for small objects, from Darknet block 5 output) ---
        # Input from Darknet block 5 (1024 channels)
        # Assuming input_shape[1] and input_shape[2] are H and W of the original input image
        h_block5, w_block5 = self.input_shape[1] // 32, self.input_shape[2] // 32
        head1_layers = self._detection_block(1024, 512, 1024, output_channels, h_block5, w_block5)
        self.head1_start_index = len(self.layers)
        for layer in head1_layers:
            self.add(layer)
        # Head 1 - YoloOutput
        # For the first head (largest scale, smallest stride), use the third set of anchors (largest anchors)
        yolo_output1 = YoloOutput(num_anchors=3, num_classes=self.num_classes, anchors=self.anchors[2], stride=32)
        self.layers.append(yolo_output1)
        self.yolo_output_layers.append(yolo_output1)
        self.head1_end_index = len(self.layers) -1

        # --- Head 2 (for medium objects, from Darknet block 4 output + upsampled head 1) ---
        # Upsample from head 1's 512-channel feature map (before final conv)
        # This requires an Upsample layer, which is not yet implemented. For now, we'll simulate the channel changes.
        # The Route layer will concatenate this with block 4's output.
        h_block4, w_block4 = self.input_shape[1] // 16, self.input_shape[2] // 16
        # Placeholder for upsampling layer: Conv2D with stride 1/2 or TransposedConv2D
        # For now, we assume the upsampled feature map has 256 channels.
        # Let's add a 1x1 conv to reduce channels from head1's 512 to 256 before upsampling (conceptual)
        self.add(Conv2D(input_shape=(512, h_block5, w_block5), num_filters=256, kernel_size=1, padding=0)) # This layer is for upsampling path
        self.add(BatchNorm2D(num_features=256))
        self.add(LeakyReLU(alpha=0.1))
        # Upsample layer would go here, outputting (256, h_block4, w_block4)
        # For now, we'll just define the route layer assuming this upsampling happened.
        # The Route layer needs to know the indices of layers to concatenate.
        # Index of Block 4 output: self.feature_map_indices['block4']
        # Index of Upsampled output: len(self.layers) -1 (the LeakyReLU above)
        self.add(Route(layers_to_route=[self.feature_map_indices['block4'], len(self.layers)-1])) # Concatenates (512 from block4) + (256 from upsample)
        
        head2_layers = self._detection_block(512 + 256, 256, 512, output_channels, h_block4, w_block4)
        self.head2_start_index = len(self.layers)
        for layer in head2_layers:
            self.add(layer)
        # Head 2 - YoloOutput
        # For the second head (medium scale, medium stride), use the second set of anchors
        yolo_output2 = YoloOutput(num_anchors=3, num_classes=self.num_classes, anchors=self.anchors[1], stride=16)
        self.layers.append(yolo_output2)
        self.yolo_output_layers.append(yolo_output2)
        self.head2_end_index = len(self.layers) -1

        # --- Head 3 (for large objects, from Darknet block 3 output + upsampled head 2) ---
        h_block3, w_block3 = self.input_shape[1] // 8, self.input_shape[2] // 8
        # Similar upsampling path from head 2's 256-channel feature map
        self.add(Conv2D(input_shape=(256, h_block4, w_block4), num_filters=128, kernel_size=1, padding=0))
        self.add(BatchNorm2D(num_features=128))
        self.add(LeakyReLU(alpha=0.1))
        # Upsample layer here
        self.add(Route(layers_to_route=[self.feature_map_indices['block3'], len(self.layers)-1])) # Concatenates (256 from block3) + (128 from upsample)

        head3_layers = self._detection_block(256 + 128, 128, 256, output_channels, h_block3, w_block3)
        self.head3_start_index = len(self.layers)
        for layer in head3_layers:
            self.add(layer)
        # Head 3 - YoloOutput
        # For the third head (smallest scale, largest stride), use the first set of anchors (smallest anchors)
        yolo_output3 = YoloOutput(num_anchors=3, num_classes=self.num_classes, anchors=self.anchors[0], stride=8)
        self.layers.append(yolo_output3)
        self.yolo_output_layers.append(yolo_output3)
        self.head3_end_index = len(self.layers) -1

        print("YOLOv3 detection heads built (Upsampling is conceptual).")

    def forward(self, input_data, training=True):
        # Store all layer outputs during the forward pass
        all_layer_outputs = [] 
        x = input_data

        # Backbone pass
        for i, layer in enumerate(self.layers[:self.feature_map_indices['block5'] + 1]): # Up to end of Darknet-53
            if isinstance(layer, (Dropout, BatchNorm2D)):
                x = layer.forward(x, training=training)
            else:
                x = layer.forward(x)
            all_layer_outputs.append(x)
        
        # Detection Head 1
        x_head1 = all_layer_outputs[self.feature_map_indices['block5']]
        for i in range(self.head1_start_index, self.head1_end_index + 1):
            layer = self.layers[i]
            if isinstance(layer, (Dropout, BatchNorm2D)):
                x_head1 = layer.forward(x_head1, training=training)
            else:
                x_head1 = layer.forward(x_head1)
            all_layer_outputs.append(x_head1) # Store output of this layer
        output1 = x_head1 # Output of YoloOutput layer for head 1

        # Detection Head 2
        # Get feature map from backbone (block4)
        fm_block4 = all_layer_outputs[self.feature_map_indices['block4']]

        # Process the upsampling path from Head 1
        # The layers for upsampling path for head 2 are added right after head1_end_index
        # These are: Conv2D, BatchNorm2D, LeakyReLU
        upsample_conv_layer = self.layers[self.head1_end_index + 1]
        upsample_bn_layer = self.layers[self.head1_end_index + 2]
        upsample_relu_layer = self.layers[self.head1_end_index + 3]

        x_upsample_path_head2 = upsample_conv_layer.forward(all_layer_outputs[self.head1_end_index - 1], training=training) # Input to this is output of 512->1024 conv in head1
        x_upsample_path_head2 = upsample_bn_layer.forward(x_upsample_path_head2, training=training)
        x_upsample_path_head2 = upsample_relu_layer.forward(x_upsample_path_head2)
        # Conceptual Upsample: For now, we assume this output is already upsampled to the correct spatial dimensions
        # (N, 256, H/16, W/16) from (N, 256, H/32, W/32) after the 1x1 conv and activation.
        # A proper Upsample layer would be needed here.
        # For demonstration, we'll resize it manually to match the target feature map size for concatenation.
        # In a real scenario, an Upsample layer would handle this.
        target_h_head2 = fm_block4.shape[2]
        target_w_head2 = fm_block4.shape[3]
        # Simple nearest neighbor upsampling (conceptual, not actual layer)
        # This is a placeholder for a proper upsampling layer.
        x_upsample_path_head2_resized = np.repeat(np.repeat(x_upsample_path_head2, 2, axis=2), 2, axis=3) # Assuming 2x upsample

        # Route for Head 2
        route_layer_head2 = self.layers[self.head1_end_index + 4] # The Route layer for head 2
        x_head2_input = route_layer_head2.forward([fm_block4, x_upsample_path_head2_resized])
        all_layer_outputs.append(x_head2_input)

        x_head2 = x_head2_input
        for i in range(self.head2_start_index, self.head2_end_index + 1):
            layer = self.layers[i]
            if isinstance(layer, (Dropout, BatchNorm2D)):
                x_head2 = layer.forward(x_head2, training=training)
            elif isinstance(layer, Route):
                # Route layer is handled explicitly before the loop
                pass
            else:
                x_head2 = layer.forward(x_head2)
            all_layer_outputs.append(x_head2)
        output2 = x_head2

        # Detection Head 3
        fm_block3 = all_layer_outputs[self.feature_map_indices['block3']]

        # Process the upsampling path from Head 2
        upsample_conv_layer_h3 = self.layers[self.head2_end_index + 1]
        upsample_bn_layer_h3 = self.layers[self.head2_end_index + 2]
        upsample_relu_layer_h3 = self.layers[self.head2_end_index + 3]

        x_upsample_path_head3 = upsample_conv_layer_h3.forward(all_layer_outputs[self.head2_end_index - 1], training=training) # Input from head2's 256->512 conv
        x_upsample_path_head3 = upsample_bn_layer_h3.forward(x_upsample_path_head3, training=training)
        x_upsample_path_head3 = upsample_relu_layer_h3.forward(x_upsample_path_head3)

        target_h_head3 = fm_block3.shape[2]
        target_w_head3 = fm_block3.shape[3]
        x_upsample_path_head3_resized = np.repeat(np.repeat(x_upsample_path_head3, 2, axis=2), 2, axis=3) # Assuming 2x upsample

        route_layer_head3 = self.layers[self.head2_end_index + 4]
        x_head3_input = route_layer_head3.forward([fm_block3, x_upsample_path_head3_resized])
        all_layer_outputs.append(x_head3_input)

        x_head3 = x_head3_input
        for i in range(self.head3_start_index, self.head3_end_index + 1):
            layer = self.layers[i]
            if isinstance(layer, (Dropout, BatchNorm2D)):
                x_head3 = layer.forward(x_head3, training=training)
            elif isinstance(layer, Route):
                pass
            else:
                x_head3 = layer.forward(x_head3)
            all_layer_outputs.append(x_head3)
        output3 = x_head3

        # Final YoloOutput layers
        # The YoloOutput layer is the last layer in each detection head block.
        # Its forward pass will handle the final processing of the feature map
        # into bounding box predictions, objectness scores, and class probabilities.
        output1 = self.layers[self.head1_end_index].forward(output1)
        output2 = self.layers[self.head2_end_index].forward(output2)
        output3 = self.layers[self.head3_end_index].forward(output3)

        return [output1, output2, output3]

    def predict(self, X):
        # This method will combine the outputs from the three detection heads
        # and apply non-maximum suppression (NMS) to get the final bounding box predictions.
        outputs = self.forward(X, training=False)

        # Placeholder for combining and post-processing outputs (e.g., NMS)
        # In a full implementation, this would involve:
        # 1. Decoding raw YOLO output to bounding box coordinates, objectness, and class probabilities.
        # 2. Filtering boxes based on objectness confidence.
        # 3. Applying Non-Maximum Suppression (NMS) to remove duplicate detections.
        
        # Decode the raw YOLO outputs from each head
        decoded_outputs = []
        for i, output in enumerate(outputs):
            # Each output corresponds to a YoloOutput layer in self.yolo_output_layers
            yolo_output_layer = self.yolo_output_layers[i]
            decoded_output = yolo_output_layer.decode_output(
                output, 
                yolo_output_layer.anchors, 
                self.img_size, 
                self.num_classes, 
                yolo_output_layer.stride
            )
            decoded_outputs.append(decoded_output)

        # Combine all decoded predictions from different scales
        all_predictions = np.concatenate(decoded_outputs, axis=1)

        # Further processing (e.g., NMS) would happen here.
        # For now, we return the combined decoded predictions.
        return all_predictions









# Example of how a CNN model might be constructed
# class SimpleCNN(Model):
#     def __init__(self, input_shape, num_classes):
#         super().__init__()
#         self.add(Conv2D(input_shape, num_filters=32, kernel_size=3, padding=1))
#         self.add(ReLU())
#         self.add(MaxPooling2D(pool_size=2, stride=2))
#         self.add(Conv2D(input_shape=(32, input_shape[1]//2, input_shape[2]//2), num_filters=64, kernel_size=3, padding=1))
#         self.add(ReLU())
#         self.add(MaxPooling2D(pool_size=2, stride=2))
#         self.add(Flatten())
#         # Calculate input size for FC layer dynamically
#         # This requires a dummy forward pass or careful calculation
#         # For simplicity, assume a fixed size or calculate based on input_shape
#         flattened_size = (input_shape[1]//4) * (input_shape[2]//4) * 64 # Example calculation
#         self.add(FullyConnected(input_size=flattened_size, output_size=num_classes))
#         self.add(Softmax()) # For classification