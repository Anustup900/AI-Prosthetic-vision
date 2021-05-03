# AI-Prosthetic-vision

Computer Vision developed a lot in the field of Generative learning , Body pose models as well 3D image
mapping . Even we can generate inbuild AR/VR by using the power AI and Computer Vision. So this project is also based on similar concept but on the forensic and prosthetic concept .First of all prosthetics are creating similar human masks that create no difference with the actual face not even from the fitting points . Facial prosthetics helps a lott in media as well security industry along with health IT in curing patients with multiple problems including disability ,mental problems as well facial de formations and so on !
Apart from Prosthetics , we are far away developed in the field of Argumented and virtual reality! We all are using automated apps as well platforms to create vision into reality .This projects of AI prosthetic vision is developed to integrate the power of computer vision , Face map generator by the light of prosthetics in the power of Face mesh algorithm.This is specially designed to help patients with cognitive impairment and to help police to extract face maps from the face sketch of criminals !

## Tech stacks used : 
```
Python 
Tensorflow & Tf Js
Dense Pose 
Detectron 2
Open CV - Face contours
```
## Tensorflow Js : 

TensorFlow.js is a very powerful library when it comes to using deep learning models directly in the browser. It includes support for a wide range of functions, covering basic machine learning, deep learning, and even model deployment. Another important feature of Tensorflow.js is the ability to use existing pre-trained models for quickly building exciting and cool applications.TensorFlow’s face landmark detection model to predict 486 3D facial landmarks that can infer the approximate surface geometry of human faces.

## Open CV Face contours :

Contours are defined as the line joining all the points along the boundary of an image that are having the same intensity. Contours come handy in shape analysis, finding the size of the object of interest, and object detection. OpenCV has findContour() function that helps in extracting the contours from the image. It works best on binary images, so we should first apply thresholding techniques, Sobel edges, etc.

# Dense Pose & Detectron 2 : 

DensePose, is Facebook’s real-time approach for mapping all human pixels of 2D RGB images to a 3D surface-based model of the body.

Research in human understanding aims primarily at localizing a sparse set of joints, like the wrists, or elbows of humans. This may suffice for applications like gesture or action recognition, but it delivers a reduced image interpretation. We wanted to go further. Imagine trying on new clothes via a photo or putting costumes on your friend’s photos. For these types of tasks, a more complete, surface-based image interpretation is required.

The DensePose project addresses this and aims at understanding humans in images in terms of such surface-based models. Learn more about the work in this blog and our CVPR 2018 paper DensePose: Dense Human Pose Estimation In The Wild.

The DensePose project introduces:

DensePose-COCO, a large-scale ground-truth dataset with image-to-surface correspondences manually annotated on 50K COCO images.
DensePose-RCNN, a variant of Mask-RCNN, to densely regress part-specific UV coordinates within every human region at multiple frames per second.

Detectron2 was built by Facebook AI Research (FAIR) to support rapid implementation and evaluation of novel computer vision research. It includes implementations for the following object detection algorithms:
```
Mask R-CNN

RetinaNet

Faster R-CNN

RPN

Fast R-CNN

TensorMask

PointRend

DensePose
```
