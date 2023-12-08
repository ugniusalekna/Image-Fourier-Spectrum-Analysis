import numpy as np


def create_ideal_filter(shape, filter_type, cutoff_radius):
    P, Q = shape

    # filter = np.zeros((P, Q))
    # print(f'P, Q = {P, Q}')
    # for u in range(P):
    #     for v in range(Q):
    #         distance = np.sqrt((u - P//2)**2 + (v - Q//2)**2)
    #         if filter_type == 'lowpass':
    #             if distance <= cutoff_radius:
    #                 filter[u, v] = 1
    #         elif filter_type == 'highpass':
    #             if distance > cutoff_radius:
    #                 filter[u, v] = 1

    filter = np.zeros((P, Q))
    U, V = np.meshgrid(np.arange(Q), np.arange(P))
    D = np.sqrt((U - Q//2)**2 + (V - P//2)**2)
    if filter_type == 'lowpass':
        filter[D <= cutoff_radius] = 1
    elif filter_type == 'highpass':
        filter[D > cutoff_radius] = 1

    return filter


def create_butterworth_filter(shape, filter_type, cutoff_radius, order=1.):
    P, Q = shape

    # filter = np.zeros((P, Q))
    # for u in range(P):
    #     for v in range(Q):
    #         distance = np.sqrt((u - P//2)**2 + (v - Q//2)**2)
    #         if filter_type == 'lowpass':
    #             filter[u, v] = 1 / ((1 + distance / cutoff_radius)**(2 * order))
    #         elif filter_type == 'highpass':
    #             filter[u, v] = 1 / ((1 + cutoff_radius / distance)**(2 * order))

    U, V = np.meshgrid(np.arange(Q), np.arange(P))
    D = np.sqrt((U - Q//2)**2 + (V - P//2)**2)
    if filter_type == 'lowpass':
        filter = 1 / ((1 + D / cutoff_radius)**(2 * order))
    elif filter_type == 'highpass':
        filter = 1 / ((1 + cutoff_radius / (D+1e-9))**(2 * order))

    return filter


def create_gaussian_filter(shape, filter_type, cutoff_radius):
    P, Q = shape

    # filter = np.zeros((P, Q))
    # for u in range(P):
    #     for v in range(Q):
    #         distance = np.sqrt((u - P//2)**2 + (v - Q//2)**2)
    #         if filter_type == 'lowpass':
    #             filter[u, v] = np.exp(-distance**2 / (2 * cutoff_radius**2))
    #         elif filter_type == 'highpass':
    #             filter[u, v] = 1 - np.exp(-distance**2 / (2 * cutoff_radius**2))

    U, V = np.meshgrid(np.arange(Q), np.arange(P))
    D = np.sqrt((U - Q//2)**2 + (V - P//2)**2)
    if filter_type == 'lowpass':
        filter = np.exp( -D**2 / (2 * cutoff_radius**2))
    elif filter_type == 'highpass':
        filter = 1 - np.exp( -D**2 / (2 * cutoff_radius**2))

    return filter


def apply_frequency_filtering(image_amplitude, filter_name, filter_type, cutoff_radius, order=1.):
    if filter_name == 'ideal':
        filter = create_ideal_filter(image_amplitude.shape, filter_type, cutoff_radius)
    elif filter_name == 'butterworth':
        filter = create_butterworth_filter(image_amplitude.shape, filter_type, cutoff_radius, order)
    elif filter_name == 'gaussian':
        filter = create_gaussian_filter(image_amplitude.shape, filter_type, cutoff_radius)
    
    filtered_image_amplitude = image_amplitude * filter

    return filtered_image_amplitude