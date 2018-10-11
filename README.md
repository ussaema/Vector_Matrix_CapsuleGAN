# Capsule Generative Adversarial Networks - Keras
> git clone https://github.com/ussaema/Vector_Matrix_CapsuleGAN.git

**contact: oussema.dhaouadi@tum.de**  
Models were trained using 2 GeForce GTX 1080 GPUs. All networks were trained using early stopping which is triggered when no improvement on the generated samples is remarkable.
An early stopping under the condition that the loss of the validation set has not been lowered in some straight epochs is not possible due to the alternation in optimising the losses of Minimax game. By cause of limitations in time and for the sake of comparison, a number of epochs has been set to 100.
It is also possible to train the discriminator (Vector CaspNet, Matrix CapsNet, etc.) as a simple classifer on MNIST, smallNORB or CIFAR10 by adding # at the beginning of the second line of the following part:
 
    "models_to_train": {DISCRIMINATOR['name'] : DISCRIMINATOR['train'],
                           GENERATOR['name']:GENERATOR['train'], 
                          },


## Generated samples
**MVCaps1 MNIST** <br />

![enter image description here](https://github.com/ussaema/Vector_Matrix_CapsuleGAN/blob/master/imgs/MVCaps1_MNIST_epoch99.png?raw=true)
<br />
**MVCaps1 smallNORB**<br />

![enter image description here](https://github.com/ussaema/Vector_Matrix_CapsuleGAN/blob/master/imgs/MVCaps1_smallNORB_epoch99.png?raw=true)<br />
**MVCaps1 CIFAR-10**<br />

![enter image description here](https://github.com/ussaema/Vector_Matrix_CapsuleGAN/blob/master/imgs/MVCaps1_CIFAR10_epoch99.png?raw=true)
<br />
## Models' configurations
![enter image description here](https://github.com/ussaema/Vector_Matrix_CapsuleGAN/blob/master/imgs/gen_archi.png?raw=true)
![enter image description here](https://github.com/ussaema/Vector_Matrix_CapsuleGAN/blob/master/imgs/disc_archi_dcgan_wgan.png?raw=true)
![enter image description here](https://github.com/ussaema/Vector_Matrix_CapsuleGAN/blob/master/imgs/disc_archi_mcapsgan.png?raw=true)
![enter image description here](https://github.com/ussaema/Vector_Matrix_CapsuleGAN/blob/master/imgs/disc_archi_vcapsgan.png?raw=true)
## Scores
![enter image description here](https://github.com/ussaema/Vector_Matrix_CapsuleGAN/blob/master/imgs/scores.png?raw=true)

Matrix CapsGAN failed to learn! Feel free to contact me and send suggestions about changes in the architecture, parameters, .... :)

#### Copyright & License

Copyright (C) 2018 Oussema Dhaouadi - TUM

Licensed under MIT License

See the  [LICENSE](https://github.com/ussaema/Vector_Matrix_CapsuleGAN/blob/master/LICENSE).
