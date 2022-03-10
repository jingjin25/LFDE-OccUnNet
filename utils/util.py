


def crop_boundary(I, crop_size):
    '''crop the boundary (the last 2 dimensions) of a tensor'''
    if crop_size == 0:
        return I

    if crop_size > 0:
        size = list(I.shape)
        I_crop = I.view(-1, size[-2], size[-1])
        I_crop = I_crop[:, crop_size:-crop_size, crop_size:-crop_size]
        size[-1] -= crop_size * 2
        size[-2] -= crop_size * 2
        I_crop = I_crop.view(size)
        return I_crop

    
 


 