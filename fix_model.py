import h5py

filename = 'fake_job_model.h5'
f = h5py.File(filename, mode='r+')
model_config_string = f.attrs.get('model_config')
if model_config_string and '"batch_shape":' in model_config_string:
    print("Found batch_shape, replacing with batch_input_shape")
    model_config_string = model_config_string.replace('"batch_shape":', '"batch_input_shape":')
    f.attrs.modify('model_config', model_config_string.encode('utf-8') if isinstance(model_config_string, str) else model_config_string)
else:
    print("No batch_shape found")

f.close()
