<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  
  <h3 align="center">SUPER OCR</h3>

  <p align="center">
    Templateless OCR solution for Maverics Botathon
    
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#links-to-our-solution">links to our solution</a></li>
    <li><a href="#references">Reference</a></li>
    <li><a href="#license">License</a></li>
   
    
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

![Product Name Screen Shot](https://github.com/RajArPatra/Super-OCR/blob/master/input%20image%20(1).png)

We approach this problem in 2 parts to obtain  the details of the Invoice.

The first Part:

First our Algorithm Uses Thresholding And Morphological Transforms to detect upper boxes and after these upper boxes are detected, the text is obtained using an OCR and then its stored in a “csv” file.

The Second Part:

For the second part we use Tablenet , a deep learning model inspired from the paper:-
After the image is passed through tablenet , tables along with columns are detected which makes it easier to get the line data from the Central Table and after this these are passed through OCR to get the Text,and then its stored in a “csv” file.Our solution is robust enough to process both digital and scanned copies of Invoices.




### Built With

This section should list any major frameworks that you built your project using. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.
* [PyTorch](http://pytorch.org/)
* [Streamlit](streamlit.io)
* [Pytesseract](https://pypi.org/project/pytesseract/)


## LINKS TO OUR SOLUTION
* [Weight file](https://drive.google.com/file/d/1Tz9Y2MaS60eTx7HVfs9jQ-ZW9E_cAvqf/view?usp=sharing)
* [Custom dataset](https://drive.google.com/file/d/1Tz9Y2MaS60eTx7HVfs9jQ-ZW9E_cAvqf/view?usp=sharing)
* [Colab file](https://colab.research.google.com/drive/1M58WvFQnr31LwGE-sceo_pauvQS39jxu?usp=sharing)
* [Video file](https://drive.google.com/file/d/1-apgY8D33nq20gM-QCwmZouoV7lVVApZ/view?usp=sharing)
* [Github](https://github.com/RajArPatra/Super-OCR)
* [Additional resources](https://drive.google.com/drive/folders/1TZHgGlCqzgw5s86DRNAVyxzjAY5h2lhU?usp=sharing)
   

## References
* [https://arxiv.org/abs/2001.01469](https://arxiv.org/abs/2001.01469)
* [https://arxiv.org/abs/1703.06870](https://arxiv.org/abs/1703.06870)
* [https://link.springer.com/chapter/10.1007/11551188_67#:~:text=We%20propose%20a%20workflow%20for,and%20(iii)%20table%20detection.](https://link.springer.com/chapter/10.1007/11551188_67#:~:text=We%20propose%20a%20workflow%20for,and%20(iii)%20table%20detection.)
* [https://arxiv.org/abs/2011.13534](https://arxiv.org/abs/2011.13534)





<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->






<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
