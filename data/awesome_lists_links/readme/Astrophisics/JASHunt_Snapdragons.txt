# Snapdragons
Snapdragons population synthesis &amp; mock survey data generation code 
Please cite the Snapdragons paper, if you use it for publications
Hunt et al. 2015, MNRAS, 450, 2132
http://adsabs.harvard.edu/abs/2015MNRAS.450.2132H

Input: N-body model.
Output: Stellar data.

The code has a Makefile in the main directory, which compiles the source code in src.
For anyone unfamiliar, to compile just type 'make' in the main directory

You must dowload the extinction maps separately from https://1drv.ms/u/s!AnMgd4KLTt9j6HtlSWaRd9kP5Rem?e=QcCqax and place them in the ini folder (not inside UBV)

--------

(If you want to increase MN to include more than ~9125000 N-body particle model you may need to add -mcmodel=medium after TGAS-errors.F)

Input in pop_synth.F is

open(30,file='ini/input.dat',status='unknown')

c *** x,y,z(kpc),vx,vy,vz(km/s),metallicity(z),age(Gyr),mass(solar)

read(30,*,end=603) x_p,y_p,z_p,vx_p,vy_p,vz_p,z,tss,m_p

Output to GeneratedStars.dat or GeneratedStars.bin depending on the flag at the beginning

c *** Output to binary (1) or ASCII (0)? ***

Binary=0

You should also set magnitude limit up to which to generate the stars:

c *** Minimum magnitude necessary ***

      ChosenLim=16.0d0

      write(6,*) 'Limited to V<‘,ChosenLim

For differnt data releases, set the error parameter in the beginning of PopSynth.F
Gaia end of mission -> error=1  
Gaia TGAS -> error=2
Gaia TGAS (Hipparcos stars only) -> error=3
Gaia DR2 -> error=4

The Attached pop_synth.F has the full range of outputs turned on, but you can adjust that by taking out fields from the below 
print statement, and editing the 102 format label to show correct length.

Let me know if you can’t get it to work, or have any questions about what’s going on in there…

jason.hunt.11 (at) ucl.ac.uk

Cheers,
Jason

Outputs are:
                                write(10,102) (a(j),j=1,6),(ao(j),j=1,6)
     &                             ,(ae(j),j=1,6),(p(j),j=1,2)
     &                             ,(po(j),j=1,2),(pe(j),j=1,2)
     &                             ,vgen,d_p,x_p,y_p,z_p,vx_p,vy_p,vz_p
     &                             ,mgen,GRVS,VIgen,ex31,avgen
     &                             ,avgen-aigen,z,PAge,od_p,xo_p,yo_p
     &                             ,zo_p,vxo_p,vyo_p,vzo_p,vro_p,vroto_p
     &                             ,rp,vr_p,vrot_p,rop,makms,mdkms
     &                             ,makmso,mdkmso,l_p,b_p,lo_p,bo_p
     &                             ,vl_p,vb_p,vlo_p,vbo_p
     &                             ,ugen,bgen,rgen,jgen,hgen,kgen
     &                             ,Ue,Ve,We

Which are: 

1 true ra
2 true dec
3 true parallax
4 true pmra
5 true pmdec
6 true rv
7 observed ra
8 observed dec
9 observed parallax
10 observed pmra
11 observed pmdec
12 observed  rv
13 ra standard deviation 
14 dec standard deviation 
15 parallax standard deviation 
16 pmra standard deviation 
17 pmdec standard deviation 
18 rv standard deviation 
19 G Magnitude 
20 G_bp - G_rp
21 observed G Magnitude 
22 observed G_bp - G_rp 
23 G Magnitude Gaia standard deviation 
24 G_bp - G_rp Gaia standard deviation 
25 apparent V mag
26 distance from sun      
27 x 
28 y 
29 z 
30 vx 
31 vy 
32 vz 
33 mass (solar mass)
34 G_RVS
35 V-Igen
36 extinction
37 absolute V mag
38 absolute V-i 
39 metallicity z*10^6
40 Age (log10years)
41 observed distance
42 observed x		
43 observed y 
44 observed z 
45 observed vx 
46 observed vy 
47 observed vz 
48 observed galactocentric vr
49 observed vrot
50 galactocentric radius
51 galactocentric vr
52 vrot
53 observed galactocentric radius
54 pmra (km/s)
55 pmdec (km/s)
56 observed pmra (km/s)
57 observed pmdec (km/s)
58 l
59 b
60 observed l
61 observed b
62 vl 
63 vb
64 observed vl
65 observed vb
66 u mag
67 b mag
68 r mag
69 j mag
70 h mag
71 k mag
72 Error on U
73 Error on V
74 error on W
75 Teff
76 logg
77 observed Teff
78 observed logg 
79 Teff uncertainty
80 logg uncertainty
