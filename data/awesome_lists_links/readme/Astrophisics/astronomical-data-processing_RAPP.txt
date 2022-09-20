# Warning!
1) Do not use IDE to debugging the program, because RAPP uses the multiprocessing module.
2) The latest version of Anaconda is recommended, as the older version may not work (the February 2020 version of Anaconda can run RAPP directly, in Linux system).
3) Since the 1.26m telescope is equatorial, the angle parameter of image rotation was set to 0, and the codes of image rotation was not written temporarily.


# Simple photometry tutorial:

## First import module:

    from module.core import RAPP

## Then create photometric object:

    rapp = RAPP(targ='xxx',
    expo_key='EXPOSE',
    date_key='DATE',)

There are three required parameters, the fits folder path, the key of the exposure, and the key of the date. These two keys need to be found in the header of fits file. After creating the rapp object, you can use it to get the following parameters:

>rapp.targ: listThe list of fits path.  
>rapp.mask: ndarr Mask image. Default is an incut ellipse.  
>rapp.bias: ndarr Bias image. Default is 0.  
>rapp.dark: ndarr Dark image. Default is 0.  
>rapp.flat: ndarr Flat image. Default is 1.  

The suggestion here is to fill in all the parameters and the full version looks like this:

    rapp = RAPP(targ=,required
                bias=,
                dark=,
                flat=,
                mask=,
                expo_key=,required
                date_key=,required
                count=,
                N=)

## Once the rapp is created, then do the information initialize:

    rapp.info_init()

This step will create rapp.info. rapp.info is a pandas.DataFrame. The structure is:(jd, (geometric radius, center of mass), star).

## After information initialized, do the match:

    rapp.match()

The match() will add rapp.shifts to the rapp object. rapp.shifts is a ndarray, datatype is complex number. Each complex number represents the shift of each image to the reference image.

## After matched, do aperture photometry:

    rapp.ap()

The ap() will add rapp.table to the rapp object. rapp.table is a dictionary. The structure is:(stars, jd, (magnitude, error)).
There are 2 important parameters in ap():
a:tuple Default is (1.2, 2.4, 3.6) (aperture ratio, background inner ratio, background outer ratio)
gain: float Default is 1.

## After aperture photometry, we can save the informathin we want:

    rapp.save(result='folder')
    rapp.draw(result='folder')

save() is used to save the csv file and graph it. draw() is used to draw the reference image and mark stars.

# Overlay photometry tutorial:

## First:

    from module import core

## Then creat rapp:

    rapp = RAPP(data='xxx',
                bias='xxx',
                dark='xxx',
                flat='xxx',
                expo_key='EXPOSE',
                date_key='DATE',
                N=1,)

Because here is overlay photometry, count USES the default value of 6, which doesn't need to be too high, for reasons explained later

## Then information initialize:

    rapp.info_init()

## Then match:

    rapp.match()

## Then here is the key point, do image combine:

    img = rapp.img_combine()

## Find stars in image:

    info = rapp.find_star(raw=img,
                          ref=True,
                          count=10)

Here's why we didn't have to provide count when we created the rapp. Because the weak stars in the combined image are definitely more obvious than the original data, you can find more stars in the combined image. And also we get a new info from the combined image.

## Then aperture photometry, but it also need to throw info to ap() as an argument:

    rapp.ap(info)

## In the later process, the difference from the simple version is on draw():

    rapp.draw(result='folder',
              ref=img,
              info0=info)

Then, the reference image will draw by combine image

## Last:

    rapp.save(result='folder')

