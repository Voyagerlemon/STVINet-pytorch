import numpy as np
from numpy.random import uniform

def gen_large_mask(img_h, img_w, n): 
    """ img_h: int, an image height
    img_w: int, an image width
    marg: int, a margin for a box starting coordinate
    p_irr: float, 0 <= p_irr <= 1, a probability of a polygonal chain mask
    min_n_irr: int, min number of segments
    max_n_irr: int, max number of segments
    max_l_irr: max length of a segment in polygonal chain
    max_w_irr: max width of a segment in polygonal chain
    
    min_n_box: int, min bound for the number of box primitives
    min_n_box: int, max bound for the number of box primitives
    min_s_box: int, min length of a box side
    max_s_box: int, max length of a box side"""   

    mask = np.ones(img_h, img_w)
    # generate polygonal chain
    if uniform(0,1) < p_irr: 
       n = uniform(min_n_irr, max_n_irr) # sample number of segments

    for _ in range(n):
       y = uniform(0, img_h) # sample a starting point
       x = uniform(0, img_w)

       a = uniform(0, 360) # sample angle
       l = uniform(10, max_l_irr) # sample segment length
       w = uniform(5, max_w_irr) # sample a segment width
      
       # draw segment starting from (x,y) to (x_,y_) using brush of width w
       x_ = x + l * sin(a)
       y_ = y + l * cos(a)
      
       gen_segment_mask(mask, start=(x, y), end=(x_, y_), brush_width=w)
       x, y = x_, y_
    else: # generate Box masks
       n = uniform(min_n_box, min_n_box) # sample number of rectangles
      
       for _ in range(n):
          h = uniform(min_s_box, max_s_box) # sample box shape
          w = uniform(min_s_box, max_s_box)
      
          x_0 = uniform(marg, img_w - marg + w) # sample upper-left coordinates of box
          y_0 = uniform(marg, img_h - marg - h)
      
          gen_box_mask(mask, size=(img_w, img_h), masked=(x_0, y_0, w, h))
    return mask