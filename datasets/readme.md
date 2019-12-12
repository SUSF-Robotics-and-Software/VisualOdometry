# Datasets

The following sets are used, and must be downloaded seperately for use in 
testing.

## Devon Island Rover Traverse

*Location*: `datasets/devon`

*Source*: http://asrl.utias.utoronto.ca/datasets/devon-island-rover-navigation/rover-traverse.html

*Instructions*:

The dataset is huge, we don't need all of it. We're just going to use the 
greyscale rectified images for sequence 00. These can be found at the link in 
the wget command below if you prefer to download them through your browser.

You then need to extract them into the devon folder

To download and setup the file higherachy on linux:
```
cd datasets
wget ftp://asrl3.utias.utoronto.ca/Devon-Island-Rover-Navigation/rover-traverse/grey-rectified-512x384/grey-rectified-512x384-s00.zip
mkdir -p devon/imgs
unzip grey-rectified-512x384-s00.zip -d devon/imgs/
cd devon
wget ftp://asrl3.utias.utoronto.ca/Devon-Island-Rover-Navigation/rover-traverse/logs/image-times.txt
wget ftp://asrl3.utias.utoronto.ca/Devon-Island-Rover-Navigation/rover-traverse/logs/gps-topocentric.txt
wget ftp://asrl3.utias.utoronto.ca/Devon-Island-Rover-Navigation/rover-traverse/logs/inclinometer-sampled.txt
```

To download on windows either run the above in a git bash shell or you can 
download the files individually.

