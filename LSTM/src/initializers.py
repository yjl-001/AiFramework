# LSTM/src/initializers.py
import numpy as np

def zeros(shape):
    return np.zeros(shape)

def ones(shape):
    return np.ones(shape)

def random_normal(shape, mean=0.0, std=0.01):
    return np.random.normal(mean, std, size=shape)

def random_uniform(shape, low=-0.01, high=0.01):
    return np.random.uniform(low, high, size=shape)

def glorot_normal(shape, fan_in=None, fan_out=None):
    """Glorot (Xavier) normal initializer."""
    if fan_in is None or fan_out is None:
        # Try to infer from shape (e.g., for dense layer weights)
        if len(shape) == 2:
            fan_in, fan_out = shape[0], shape[1]
        elif len(shape) > 2: # For conv layers, etc. - more complex
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        else:
            fan_in = fan_out = int(np.sqrt(np.prod(shape)))
            # raise ValueError("Cannot infer fan_in and fan_out for Glorot normal initializer from shape {shape}")

    std = np.sqrt(2.0 / (fan_in + fan_out))
    return random_normal(shape, std=std)

def glorot_uniform(shape, fan_in=None, fan_out=None):
    """Glorot (Xavier) uniform initializer."""
    if fan_in is None or fan_out is None:
        if len(shape) == 2:
            fan_in, fan_out = shape[0], shape[1]
        elif len(shape) > 2:
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        else:
            fan_in = fan_out = int(np.sqrt(np.prod(shape)))
            # raise ValueError("Cannot infer fan_in and fan_out for Glorot uniform initializer from shape {shape}")

    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return random_uniform(shape, low=-limit, high=limit)

def he_normal(shape, fan_in=None):
    """He normal initializer."""
    if fan_in is None:
        if len(shape) == 2:
            fan_in = shape[0]
        elif len(shape) > 2:
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
        else:
            fan_in = int(np.sqrt(np.prod(shape)))
            # raise ValueError("Cannot infer fan_in for He normal initializer from shape {shape}")

    std = np.sqrt(2.0 / fan_in)
    return random_normal(shape, std=std)

def he_uniform(shape, fan_in=None):
    """He uniform initializer."""
    if fan_in is None:
        if len(shape) == 2:
            fan_in = shape[0]
        elif len(shape) > 2:
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
        else:
            fan_in = int(np.sqrt(np.prod(shape)))
            # raise ValueError("Cannot infer fan_in for He uniform initializer from shape {shape}")

    limit = np.sqrt(6.0 / fan_in)
    return random_uniform(shape, low=-limit, high=limit)

INITIALIZERS = {
    'zeros': zeros,
    'ones': ones,
    'random_normal': random_normal,
    'random_uniform': random_uniform,
    'glorot_normal': glorot_normal,
    'glorot_uniform': glorot_uniform,
    'he_normal': he_normal,
    'he_uniform': he_uniform,
}

def get_initializer(name):
    func = INITIALIZERS.get(name.lower())
    if func is None:
        raise ValueError(f"Initializer '{name}' not found. Available: {list(INITIALIZERS.keys())}")
    return func

if __name__ == '__main__':
    shape_dense = (100, 200)
    shape_conv = (32, 3, 3, 3) # filters, channels, height, width

    print(f"zeros(shape_dense):\n{zeros(shape_dense)[:2,:2]}...")
    print(f"ones(shape_dense):\n{ones(shape_dense)[:2,:2]}...")
    print(f"random_normal(shape_dense):\n{random_normal(shape_dense)[:2,:2]}...")
    print(f"random_uniform(shape_dense):\n{random_uniform(shape_dense)[:2,:2]}...")
    
    print(f"glorot_normal(shape_dense):\n{glorot_normal(shape_dense)[:2,:2]}...")
    print(f"glorot_uniform(shape_dense):\n{glorot_uniform(shape_dense)[:2,:2]}...")
    print(f"he_normal(shape_dense):\n{he_normal(shape_dense)[:2,:2]}...")
    print(f"he_uniform(shape_dense):\n{he_uniform(shape_dense)[:2,:2]}...")

    # Test with fan_in/fan_out provided
    print(f"glorot_normal with explicit fan_in/out:\n{glorot_normal(shape_dense, fan_in=100, fan_out=200)[:2,:2]}...")
    print(f"he_normal with explicit fan_in:\n{he_normal(shape_dense, fan_in=100)[:2,:2]}...")

    # Test for convolutional shape (example)
    # Note: fan_in/fan_out for conv layers can be defined in multiple ways.
    # This is a common way for Keras-like interpretation.
    print(f"glorot_normal(shape_conv):\n{glorot_normal(shape_conv)[0,0,:2,:2]}...")
    print(f"he_normal(shape_conv):\n{he_normal(shape_conv)[0,0,:2,:2]}...")

    try:
        get_initializer('zeros')
        print("Initializer getter OK.")
        get_initializer('unknown')
    except ValueError as e:
        print(f"Caught expected error: {e}")