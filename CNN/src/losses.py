import numpy as np

class Loss:
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    def gradient(self, y_true, y_pred):
        raise NotImplementedError

class MeanSquaredError(Loss):
    def loss(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def gradient(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

class CategoricalCrossentropy(Loss):
    def loss(self, y_true, y_pred):
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    def gradient(self, y_true, y_pred):
        # Clip predictions to avoid division by zero
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -(y_true / y_pred) / y_true.shape[0]

class BinaryCrossentropy(Loss):
    def loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def gradient(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / y_true.size


class YOLOLoss(Loss):
    def __init__(self, anchors, img_size, num_classes, ignore_thresh=0.5):
        self.anchors = anchors
        self.img_size = img_size
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.mse_loss = MeanSquaredError()
        self.bce_loss = BinaryCrossentropy()

    def loss(self, y_true, y_pred_raw, stride):
        # y_true: (batch_size, grid_h, grid_w, num_anchors, 5 + num_classes)
        # y_pred_raw: (batch_size, grid_h, grid_w, num_anchors * (5 + num_classes))

        batch_size, grid_h, grid_w, _ = y_pred_raw.shape
        num_anchors = len(self.anchors)

        # Reshape y_pred_raw to (batch_size, grid_h, grid_w, num_anchors, 5 + num_classes)
        y_pred = y_pred_raw.reshape(batch_size, grid_h, grid_w, num_anchors, 5 + self.num_classes)

        # Get components from y_pred
        pred_x = sigmoid(y_pred[..., 0])
        pred_y = sigmoid(y_pred[..., 1])
        pred_w = y_pred[..., 2]
        pred_h = y_pred[..., 3]
        pred_obj = sigmoid(y_pred[..., 4])
        pred_cls = sigmoid(y_pred[..., 5:])

        # Get components from y_true
        true_x = y_true[..., 0]
        true_y = y_true[..., 1]
        true_w = y_true[..., 2]
        true_h = y_true[..., 3]
        true_obj = y_true[..., 4]
        true_cls = y_true[..., 5:]
        # Mask for responsible anchor (where true_obj == 1)
        obj_mask = true_obj.astype(bool)
        noobj_mask = (1 - true_obj).astype(bool)

        # Create grid for x, y offsets
        grid_x = np.arange(grid_w).reshape(1, 1, grid_w, 1)
        grid_y = np.arange(grid_h).reshape(1, grid_h, 1, 1)

        # Calculate predicted bounding box coordinates in image scale
        # (x + grid_x) * stride
        # (y + grid_y) * stride
        # exp(w) * anchor_w
        # exp(h) * anchor_h
        scaled_anchors = np.array(self.anchors) / stride
        anchor_w = scaled_anchors[:, 0].reshape(1, 1, 1, num_anchors)
        anchor_h = scaled_anchors[:, 1].reshape(1, 1, 1, num_anchors)

        pred_boxes_x = (pred_x + grid_x) * stride
        pred_boxes_y = (pred_y + grid_y) * stride
        pred_boxes_w = np.exp(pred_w) * anchor_w * stride
        pred_boxes_h = np.exp(pred_h) * anchor_h * stride

        # Convert true boxes to network output format (inverse of decode_output)
        # tx = true_x / stride - grid_x
        # ty = true_y / stride - grid_y
        # tw = log(true_w / anchor_w)
        # th = log(true_h / anchor_h)
        target_x = true_x / stride - grid_x
        target_y = true_y / stride - grid_y
        target_w = np.log(true_w / (anchor_w * stride) + 1e-16)
        target_h = np.log(true_h / (anchor_h * stride) + 1e-16)

        # Bounding Box Loss (only for responsible anchors)
        loss_x = self.mse_loss.loss(target_x[obj_mask], pred_x[obj_mask])
        loss_y = self.mse_loss.loss(target_y[obj_mask], pred_y[obj_mask])
        loss_w = self.mse_loss.loss(target_w[obj_mask], pred_w[obj_mask])
        loss_h = self.mse_loss.loss(target_h[obj_mask], pred_h[obj_mask])
        bbox_loss = loss_x + loss_y + loss_w + loss_h

        # Objectness Loss
        # For responsible anchors (obj_mask), we want pred_obj to be 1
        obj_loss = self.bce_loss.loss(true_obj[obj_mask], pred_obj[obj_mask])

        # For non-responsible anchors (noobj_mask), we want pred_obj to be 0
        # But only if the predicted box doesn't have a high IoU with any ground truth box
        # This part requires calculating IoU for all predicted boxes with all ground truth boxes
        # and setting a noobj_mask based on ignore_thresh. This is complex for a simple loss class.
        # For simplicity, we'll apply noobj_mask directly based on true_obj for now.
        # A more complete implementation would involve IoU calculation and dynamic noobj_mask.
        noobj_loss = self.bce_loss.loss(true_obj[noobj_mask], pred_obj[noobj_mask])

        # Classification Loss (only for responsible anchors)
        class_loss = self.bce_loss.loss(true_cls[obj_mask], pred_cls[obj_mask])

        total_loss = bbox_loss + obj_loss + noobj_loss + class_loss

        return total_loss

    def gradient(self, y_true, y_pred_raw, stride):
        # This is a simplified gradient calculation. A full YOLO gradient is very complex.
        # It would involve backpropagating through all the sigmoid, exp, and loss functions.
        # For a custom framework, this would typically be handled by the layers themselves
        # and the loss function would provide the initial gradient to the last layer.

        # For now, we'll return a dummy gradient or raise NotImplementedError.
        # The gradient calculation for YOLO loss is usually integrated with the training loop
        # and the backpropagation of the model.
        raise NotImplementedError("YOLO loss gradient is complex and typically handled by the training loop.")