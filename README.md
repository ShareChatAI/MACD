# MACD

## Training the models:

```
bash train.sh
```
Please set the ```language``` and ```model_name``` in ```train.sh``` for finetuning the model for the desired language.

## Dataset:

```dataset``` folder contains the <em>training, validation and test</em> split in csv format for all the languages 
included with <strong>MACD</strong> - <em>Hindi, Tamil, Telugu, Malayalam and Kannada</em>.

Each row in these files contains the label and text where labels 0 and 1 are used for <strong>abusive</strong> and <strong>non-abusive</strong> comments, respectively.


## Installation:

```
conda env create -f environment.yml
```

