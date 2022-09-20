 validation
 
 **in codes/vsuite**

   code to compare several observations to simulated data with stellar mass and star formation rate, plus required data files (for observations)

   generates 7 plots 

4 plots are stellar mass functions:

all, quiescent, star forming, and all 3 on one page, compared to several observations described below.
(quiescent/star forming division at log sfr = -0.49 + (0.65+slopeval) (logM* - 10) +1.07 *(z-0.1)+shiftval.  This is PRIMUS separation for slopeval=shiftval=0, i.e. Moustakas et al, 2013,  eq 2). many of the datasets use UVJ and are plotted as is for their star forming and quiescent galaxies.

1 plot is stellar mass-sfr diagram [can be compared with e.g., Moustakas et al 2013, but not overplotted with it]

1 plot is ssfr in 4 stellar mass bins* (no cut on ra, dec for this)

1 plot is stellar mass to halo mass diagram, compared to Behroozi, Wechsler, Conroy 2013 and Moster, Naab, White 2013 fits.
Behroozi,Wechsler,Conroy 2013 use Mvir and
Moster, Naab, White 2013 use M200

More info available at the top of valid_suite.py and in the appendix of Cohn 2016, http://arxiv.org/abs/1609.03956

 
 in **codes/StellarMass/**
 
   code to compare simulated data stellar mass function with observed
   stellar mass function from PRIMUS or SDSS-GALEX, in several redshift bins from 0.01-1.0
   makes plot with both on it
  
   2 files plus your simulated data are needed:
   
     PRIMUS_stellar.py 
     
         histograms an input simulated data set (M*, sfr) with
         Moustakas et al 2013 PRIMUS data, redshifts 0.2-1.0.
         Moustakas et al 2013 SDSS-GALEX data, redshifts 0.01-0.2
         
     Mous_13_table4.txt 
     
         is 2013 PRIMUS data, in same directory 
         (typed in table 4 of Moustakas et al 2013, arXiv:1301.1688v1)

 in **codes/Bband/**
 
   code to compare simulated data B band luminosity function with observed
   stellar mass function from BOOTS, in several redshift bins from 0.2-1.1
   makes plots of both together.
   
   **only tested on 'fake' simulated data so far**
   
   2 files plus your simulated data are needed:
   
     BOOTES_lf.py 
     
         histograms an input simulated data set (M_B, M_U) with
         Beare et al 2015 BOOTES data, redshifts 0.2-1.1.
         
     Beare_tab8910.txt
     
         is 2015 BOOTES data, in same directory 
         (typed in table 8,9,10 of Beare et al 2015, arXiv:1511.01580)

