import tensorflow as tf
import hls4ml
import yaml

# Init simple Keras model
inputs = tf.keras.Input(shape=(16,), name="input_layer")
x = tf.keras.layers.Dense(8, activation='relu', name='dense')(inputs)
outputs = tf.keras.layers.Dense(4, activation='softmax', name='dense_out')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Init hls4ml config
config = hls4ml.utils.config_from_keras_model(model, granularity='name')

# Layer debugging
print("Layer keys in config:", list(config['LayerName'].keys()))

# Find softmax layer name
softmax_names = [layer.name for layer in model.layers
                 if hasattr(layer, "activation") and getattr(layer.activation, "__name__", "") == "softmax"]
if not softmax_names:
    raise RuntimeError("No softmax layer found in model")
softmax_layer_name = softmax_names[0]
print("Detected softmax layer:", softmax_layer_name)


config['Model']['Strategy'] = 'Resource'

# Softmax-layer tweaks
# LUT reduction
softmax_cfg = {
    'Precision': 'ap_fixed<16,6>',
    'exp_table_t': 'ap_fixed<18,8>',
    'inv_table_t': 'ap_fixed<18,8>',
    'ReuseFactor': 32,    
    'Strategy': 'Stable',
    'TableSize': 1024    
}
config['LayerName'][softmax_layer_name].update(softmax_cfg)

# Adjustment of reuse factor and precision for Dense layers
for lname in list(config['LayerName'].keys()):
    if 'dense' in lname.lower():
        config['LayerName'][lname]['ReuseFactor'] = 8         # 8 is a reasonable start (not 64)
        config['LayerName'][lname]['Precision'] = 'ap_fixed<12,4>'


with open('my_hls_config.yaml', 'w') as f:
    yaml.dump(config, f)
print("Wrote my_hls_config.yaml (inspect/edit if needed)")

#Conversion to HLS
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir='my_hls_prj',
    backend='Vitis'
)


print(model.to_json(indent=2))

hls_model.build(csim=True, synth=True, cosim=False, export=False)
