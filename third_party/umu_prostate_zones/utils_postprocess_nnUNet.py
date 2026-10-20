import os
import numpy as np
import skimage
from scipy.ndimage import distance_transform_edt


def get_filenames_with_extension(folder_path, extension):
    """
    Get a list of filenames with a specific extension in a folder.

    Args:
        folder_path (str): Path to the folder.
        extension (str): Desired file extension (e.g., '.nrrd').

    Returns:
        list: List of filenames with the specified extension.
    """
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(extension.lower()):
            filenames.append(filename)
    return filenames


def channelSplit(array, channel):
    return array[:, :, :, channel]


def getLargestCC(segmentation):
    labels = skimage.measure.label(segmentation, connectivity=2)
    if labels.max() != 0: # Assuming at least one CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    else:
        largestCC = labels == 100
    return largestCC


def drawUrethra(array_shape, indicies, radius_mm, inplane_image_spacing):

    new_array = np.zeros(shape=array_shape)

    radius = radius_mm / inplane_image_spacing

    # Create a square of pixels with side lengths equal to radius of circle
    x_square_min = indicies[0]-radius
    x_square_max = indicies[0]+radius+1
    y_square_min = indicies[1]-radius
    y_square_max = indicies[1]+radius+1

    # Clamp this square to the edges of the array so circles near edges don't wrap around
    if x_square_min < 0:
        x_square_min = 0
    if y_square_min < 0:
        y_square_min = 0
    if x_square_max > array_shape[0]:
        x_square_max = array_shape[0]
    if y_square_max > array_shape[1]:
        y_square_max = array_shape[1]

    # Now loop over the box and draw circle inside of it
    for p in range(int(x_square_min) , int(x_square_max)):
        for q in range(int(y_square_min) , int(y_square_max)):
            if (p - indicies[0]) ** 2 + (q - indicies[1]) ** 2 <= radius ** 2:
                new_array[p,q] = 1

    return new_array


def findIndices_allSlices(array):

    maximum = np.max(array, axis=None)

    list_of_indices = list(zip(*np.where(array == maximum)))

    plane_ind = np.round(np.average(np.asarray(list_of_indices, dtype=np.int32), axis=0))

    indices = plane_ind[:2]

    return np.append(indices, maximum, axis=None)


def column(matrix, i):
    return [row[i] for row in matrix]


def fill_empty_voxels(background, pz, cz, tz, afs, urethra, spacing):

    pz_lcc = getLargestCC(pz)
    cz_lcc = getLargestCC(cz)
    tz_lcc = getLargestCC(tz)
    afs_lcc = getLargestCC(afs)

    pz_within_prostate = pz_lcc & (~urethra.astype(bool))
    cz_within_prostate = cz_lcc & (~urethra.astype(bool))
    tz_within_prostate = tz_lcc & (~urethra.astype(bool))
    afs_within_prostate = afs_lcc & (~urethra.astype(bool))

    # Get indices of empty voxels within the prostate boundary
    empty_voxel_indices = np.argwhere((background == False) & (urethra == False) & (pz_within_prostate == False) &
                                      (cz_within_prostate == False) & (tz_within_prostate == False) &
                                      (afs_within_prostate == False))

    # Create distance maps for each zone using Euclidean distance transform
    pz_dist = distance_transform_edt(~pz_within_prostate, sampling=spacing)
    cz_dist = distance_transform_edt(~cz_within_prostate, sampling=spacing)
    tz_dist = distance_transform_edt(~tz_within_prostate, sampling=spacing)
    afs_dist = distance_transform_edt(~afs_within_prostate, sampling=spacing)

    # Iterate over empty voxel indices and assign them to the nearest zone
    for voxel_index in empty_voxel_indices:
        distances = [pz_dist[tuple(voxel_index)], cz_dist[tuple(voxel_index)], tz_dist[tuple(voxel_index)], afs_dist[tuple(voxel_index)]]
        # Find the index of the closest zone
        closest_zone_index = np.argmin(distances)
        # Assign the voxel to the closest zone
        if closest_zone_index == 0:
            pz_within_prostate[tuple(voxel_index)] = True
        elif closest_zone_index == 1:
            cz_within_prostate[tuple(voxel_index)] = True
        elif closest_zone_index == 2:
            tz_within_prostate[tuple(voxel_index)] = True
        else:
            afs_within_prostate[tuple(voxel_index)] = True


    return np.stack([background, pz_within_prostate.astype(int), cz_within_prostate.astype(int), tz_within_prostate.astype(int), afs_within_prostate.astype(int),
                     urethra], axis=-1)
