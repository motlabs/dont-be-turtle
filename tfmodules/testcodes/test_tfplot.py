"""
    testcodes for tfplot
    https://github.com/wookayin/tensorflow-plot
    2018 8 by jaewook kang

"""
import tfplot
import tfplot.summary
import tensorflow as tf
tf.InteractiveSession()

import numpy as np
from PIL import Image
import cv2
import scipy

import skimage
import skimage.data

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams, rc
from IPython.core.pylabtools import figsize
import IPython.display


def execute_plot_op(image):
    print (">>> " + str(image))
    ret = image.eval()
    plt.imshow(ret)
    plt.show()
    plt.close()

    if len(ret.shape) == 3:
        # single image
        return Image.fromarray(ret)
    elif len(ret.shape) == 4:
        return [Image.fromarray(r) for r in ret]
    else:
        raise ValueError("Invalid rank : %d" % len(ret.shape))




def execute_and_extract_image_summary(summary_op):
    from io import BytesIO

    # evaluate and extract PNG from the summary protobuf
    s = tf.Summary()
    s.ParseFromString(summary_op.eval())
    ims = []

    for i in range(len(s.value)):
        png_string = s.value[i].image.encoded_image_string
        im = Image.open(BytesIO(png_string))
        ims.append(im)

    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(ims[0])
    plt.title('imshow for batch 1')

    plt.subplot(1,2,2)
    plt.imshow(ims[1])
    plt.title('imshow for batch 2')
    plt.show()
    plt.close()

    if len(ims) == 1:
        return ims[0]
    else:
        return ims





def fake_attention():
    import scipy.ndimage
    attention = np.zeros([16, 16], dtype=np.float32)
    attention[(12, 8)] = 1.0
    attention[(10, 9)] = 1.0
    attention = scipy.ndimage.filters.gaussian_filter(attention, sigma=1.5)
    return attention

def fake_attention2():
    import scipy.ndimage
    attention = np.zeros([16, 16], dtype=np.float32)
    attention[(3, 13)] = 1.0
    attention[(4, 5)] = 1.0
    attention = scipy.ndimage.filters.gaussian_filter(attention, sigma=1.5)
    return attention


'''
    testcodes to generate sample images and its attentions
'''
sample_image = skimage.data.chelsea()
sample_image2 = skimage.data.coffee()
sample_image2 = cv2.resize(sample_image2,
                           dsize=(sample_image.shape[1],
                                  sample_image.shape[0]),
                           interpolation=cv2.INTER_CUBIC)

attention_map = fake_attention()
attention_map2 = fake_attention2()

# display the data
fig = plt.figure(1)
plt.subplot(2, 2, 1)
plt.imshow(sample_image)
plt.title('image')

plt.subplot(2, 2, 2)
plt.imshow(sample_image2)
plt.title('image2')

plt.subplot(2, 2, 3)
plt.imshow(attention_map, cmap='jet')
plt.title('attention')

plt.subplot(2, 2, 4)
plt.imshow(attention_map2, cmap='jet')
plt.title('attention2')
plt.show()


'''
an simple example for using tfplot
'''
# the input to plot_op
image_tensor = tf.constant(sample_image, name='image')
image_tensor2 = tf.constant(sample_image2, name='image')

attention_tensor = tf.constant(attention_map, name='attention')
attention_tensor2 = tf.constant(attention_map2, name='attention')

print(image_tensor)
print(attention_tensor)


def figure_attention(attention):
    fig, ax = tfplot.subplots(figsize=(4,3))
    im      = ax.imshow(attention)

    return fig

plot_op = tfplot.plot(figure_attention,[attention_tensor])
execute_plot_op(plot_op)

'''
An overlay example by using tfplot
'''

def overlay_attention(attention, image,
                      alpha=0.5, cmap='jet'):
    fig = tfplot.Figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    # fig.subplots_adjust(0, 0, 1, 1)  # get rid of margins

    H, W = attention.shape
    ax.imshow(image, extent=[0, H, 0, W])
    ax.imshow(attention, cmap=cmap,
              alpha=alpha, extent=[0, H, 0, W])
    return fig

plot_op = tfplot.plot(overlay_attention, [attention_tensor, image_tensor])
execute_plot_op(plot_op)

summary_op = tfplot.summary.plot("heatmap_summary",
                                 overlay_attention,[attention_tensor,image_tensor])

'''
    execute_and_extract_image_summary(summary_op)
    summary with batch
'''


attention_tensor_batch = tf.stack([attention_tensor,
                                   attention_tensor2],axis=0)

image_tensor_batch     = tf.stack([image_tensor,
                                   image_tensor2],axis=0)


def overlay_attention_batch(attention, image,
                            alpha=0.5, cmap='jet'):

    fig = tfplot.Figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    fig.subplots_adjust(0, 0, 1, 1)  # get rid of margins

    print (attention.shape)
    print ('[tfplot] attention  =%s' % attention)
    print ('[tfplot] image      =%s' % image)

    H, W = attention.shape
    ax.imshow(image, extent=[0, H, 0, W])
    ax.imshow(attention, cmap=cmap,
              alpha=alpha, extent=[0, H, 0, W])

    return fig



plot_op = tfplot.plot(overlay_attention_batch,
                      [attention_tensor_batch, image_tensor_batch])


summary_op = tfplot.summary.plot_many("batch_attentions_summary", overlay_attention_batch,
                                      [attention_tensor_batch, image_tensor_batch], max_outputs=2)

images = execute_and_extract_image_summary(summary_op)
