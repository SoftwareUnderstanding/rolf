# Twitter US Airline Sentiment Analysis using ULMFiT
This notebook fine-tunes a forward and backward langauge ULMFiT model on the Twitter US Airline Sentiment dataset. I have used the techniques described in https://arxiv.org/pdf/1801.06146.pdf and https://course.fast.ai/videos/?lesson=12 as the main reference for this sentiment analysis.

These techniques include:
<ul>
<li>Discriminative fine-tuning</li>
<li>Slanted triangular learning rates</li>
<li>Gradual unfreezing </li>
</ul>
I have cited my sources for the techniques applied along with my reasoning for each step within the text blocks preceeding the code implemented.
