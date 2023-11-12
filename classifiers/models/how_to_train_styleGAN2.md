
We follow the reference: https://github.com/lucidrains/stylegan2-pytorch 

Step-by-step work on command line:

 - Install the library and pytorch-fid
  >`` pip install stylegan2_pytorch pytorch-fid``
 
 - Specifiy project name and train: 
 
 > ``stylegan2_pytorch --data /path/to/images --name my-project-name --results_dir /path/to/results/dir --models_dir /path/to/models/dir --calculate-fid-every 5000``

Here path should be adjusted to each label of our classes, so at the end we will have 3 GAN model.

> We also need to see some similarity scores in cross-classes (to avoid false friends.)

 - Once the training finished, generate the minority class (scc in multiclass).

> ``$ stylegan2_pytorch  --generate``