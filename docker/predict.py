import os
import SimpleITK as sitk
from scripts.densenet_regression import DenseNet
from scripts.unet import get_unet_2D
from scripts.preprocess_utils import load_nii, enhance_noN4
from settings import target_size_unet, unet_classes, scaling_factor
import subprocess

def predict_itmt(age, gender, img_path, path_to, model_weight_path_selection, model_weight_path_segmentation, df_centile_boys_csv, df_centile_girls_csv, **kwargs):
    # Ensure the paths are properly formatted
    new_path_to = os.path.join(path_to, os.path.basename(img_path).split('.')[0])
    if not os.path.exists(new_path_to):
        os.makedirs(new_path_to)

    # Register the image
    golden_file_path = select_template_based_on_age(age)
    register_to_template_cmd(img_path, new_path_to, golden_file_path, "registered.nii.gz", create_subfolder=False)

    # Enhance and normalize the image
    image_sitk = sitk.ReadImage(os.path.join(new_path_to, "registered.nii.gz"))
    image_array = sitk.GetArrayFromImage(image_sitk)
    image_array = enhance_noN4(image_array)
    image3 = sitk.GetImageFromArray(image_array)

    sitk.WriteImage(image3, os.path.join(new_path_to, "no_z", "registered_no_z.nii"))
    cmd_line = "zscore-normalize {} -o {}".format(os.path.join(new_path_to, "no_z", "registered_no_z.nii"), os.path.join(new_path_to, "registered_z.nii"))
    subprocess.getoutput(cmd_line)

    # Load models
    model_selection = DenseNet(img_dim=(256, 256, 1), nb_layers_per_block=12, nb_dense_block=4, growth_rate=12, nb_initial_filters=16, compression_rate=0.5, sigmoid_output_activation=True, activation_type='relu', initializer='glorot_uniform', output_dimension=1, batch_norm=True)
    model_selection.load_weights(model_weight_path_selection)
    model_unet = get_unet_2D(unet_classes, (target_size_unet[0], target_size_unet[1], 1), num_convs=2, activation='relu', compression_channels=[16, 32, 64, 128, 256, 512], decompression_channels=[256, 128, 64, 32, 16])
    model_unet.load_weights(model_weight_path_segmentation)

    # Read the z-scored image
    image_sitk = sitk.ReadImage(os.path.join(new_path_to, "registered_z.nii"))
    windowed_images = sitk.GetArrayFromImage(image_sitk)

    # Resize and predict
    resize_func = functools.partial(resize, output_shape=model_selection.input_shape[1:3], preserve_range=True, anti_aliasing=True, mode='constant')
    series = np.dstack([resize_func(im) for im in windowed_images])
    series = np.transpose(series[:, :, :, np.newaxis], [2, 0, 1, 3])

    series_n = []
    for slice_idx in range(2, np.shape(series)[0] - 2):
        im_array = np.zeros((256, 256, 1, 5))
        for i in range(-2, 3):
            im_array[:, :, :, i + 2] = series[slice_idx + i, :, :, :].astype(np.float32)
        im_array = np.max(im_array, axis=3)
        series_n.append(im_array)
    series_w = np.dstack([funcy(im) for im in series_n])
    series_w = np.transpose(series_w[:, :, :, np.newaxis], [2, 0, 1, 3])

    predictions = model_selection.predict(series_w)
    slice_label = get_slice_number_from_prediction(predictions)

    image_array, affine = load_nii(os.path.join(new_path_to, "registered_z.nii"))
    infer_seg_array_3d_1 = np.zeros(image_array.shape)
    infer_seg_array_3d_2 = np.zeros(image_array.shape)

    # Rescale for the UNET
    image_array_2d = rescale(image_array[:, 15:-21, slice_label], scaling_factor).reshape(1, target_size_unet[0], target_size_unet[1], 1)
    img_half_11 = np.concatenate((image_array_2d[:, :256, :, :], np.zeros_like(image_array_2d[:, :256, :, :])), axis=1)
    img_half_12 = np.concatenate((np.zeros_like(image_array_2d[:, 256:, :, :]), image_array_2d[:, 256:, :, :]), axis=1)

    list_of_left_muscle = [img_half_11]
    list_of_right_muscle = [img_half_12]

    list_of_left_muscle_preds = []
    list_of_right_muscle_preds = []

    for image in list_of_left_muscle:
        infer_seg_array = model_unet.predict(image)
        muscle_seg = infer_seg_array[:, :, :, 1].reshape(1, target_size_unet[0], target_size_unet[1], 1)
        list_of_left_muscle_preds.append(muscle_seg)

    for image in list_of_right_muscle:
        infer_seg_array = model_unet.predict(image)
        muscle_seg = infer_seg_array[:, :, :, 1].reshape(1, target_size_unet[0], target_size_unet[1], 1)
        list_of_right_muscle_preds.append(muscle_seg)

    # Further processing and calculation...

    return # appropriate return value
