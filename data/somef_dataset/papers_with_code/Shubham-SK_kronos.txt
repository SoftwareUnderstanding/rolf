# Kronos
A voice powered healthcare app.

_created by: Vishakh Arora, Shubham Kumar, Shlok Mehrotra, and Pranav Srisankar_

___
### Problem we wish to solve
Often, when one walks into a doctor, more time is spent documenting the visit rather than giving the patient quality care. In fact, about 6 hours in an 11 hour workday is spent entering information in the [Electronic Health Records](https://en.wikipedia.org/wiki/Electronic_health_record). This leads to increased burnouts in doctor, as reported in a study that found that more than half of all doctors report experiencing at least one symptom of burnout. To reduce stress as well as increase quality of patient care, the use of scribes has risen. However, there are not nearly enough scribes for all doctors, so they are forced to share their attention with both a computer and the patient. For a patient to have the best care possible, they need the complete attention of their doctor. Kronos aims to solve this problem

### Solution
Instead of having a scribe, the doctor can transcribe their speech into a text document using Kronos. It will then use text analysis to summarize the visit, sorting the contents into prescriptions, procedures, etc.

___
### Usage
Clone the repository, navigate to the server.py file.

```
git clone https://github.com/Shubham-SK/kronos.git
cd kronos
```

Ensure [Flask](https://pypi.org/project/Flask/) is installed and run the server. It should be open [here](127.0.0.1/5000).

```
python server.py
```

Upon opening the link, you should see a greeting with your name and an option to start recording the patient-doctor conversation! When you're ready to start the checkup click record and make sure the doctor is first to speak.

### Hardware Required
Raspberry Pi with a microphone (tuned frequency so Google speech API can extrapolate). We use a lightweight [MQTT](http://mqtt.org/) messaging protocol to transmit the recording file from the Pi to a webserver where the data will be parsed and distributed to the end user.

___
### Parts of Our Project
- Data Acquisition
- NLP speech to text
- Text analysis with a Bidirectinal RNN/LSTM with Attention
- UX/UI for doctor

#### Data Acquisition
We decided to use a Raspberry Pi board to run an MQTT protocol, which can successfully transfer recording files. The recording is saved locally for the speech to text parser and for the ML model to summarize. There are two communication channels between the Raspberry Pi and computer over MQTT: one to communicate to the Raspberry Pi when to start/stop recording the conversation, and one to transfer the conversation audio file. We found great audio quality in our voice recordings, resulting in a high rate with regards to the accuracy of our translation. The use of a Raspberry Pi also allows for the maximum space optimzation due to its small footprint, resulting in its ability to comfortably be places on the doctor's desk.

#### NLP speech to text
We decided to use [Google Cloud's](https://cloud.google.com/speech-to-text/) speech to text API to convert our recordings into textual data. From our experimentation, it outperformed our expectations and showed accuracy in even converting the most complex medicine names. We designed 2 different scripts for analysis since shortly after creating the first one, we realized we had to distinguish the voices of the patient and the doctor. Luckily, google had a service for this and we were easily able to accomodate changes. Our program was able to differentiate the 2 different voices and log an entire script with respectable accuracy. This script would then be subject to summarization, feature extraction (logging, prescriptions, etc.) and will finally displayed on the wesbite.

#### Text analysis with Bidirectional RNN/LSTM with Attention
In order to make the transcript easier to read and retain essential content, we decided to take advantage of the latest state-of-the-art models to summarize our text. We researched different trending techniques ranging from Bahdanau and Luong Attention to Bidirectionall RNNs to determine the best summarization algorithm. While it was tough to find speech data, we finally curated a small dataset. After training the model on hours of patient-doctor data, we were able to obtain a very reliable summarizer. While there is plenty of room for improvement, we were proud of the accuracy it achieved. We used pytorch to define the model, train and test it. We had it training on an EC2 instance running on AWS cloud for several hours before we received a low loss.

Paper References:
- https://arxiv.org/pdf/1409.0473.pdf // Bahdanau Attention
- https://arxiv.org/pdf/1508.04025.pdf // Luong Attention
- https://papers.nips.cc/paper/5651-bidirectional-recurrent-neural-networks-as-generative-models.pdf // BDRNNs as Generative Models

#### UX/UI for doctor
We used bootstrap to construct a minimalist UI that may lack the striking looks, but does not cut a corner in functionality. We were able to accomplish everything we planned on with the webapp we created. It includes all the features required to successfully deploy in a hospital.

Screenshots of the application below:
<img style = "text-align: center" src="assets/img.png" alt="Figure 1: Webapp Dashboard Screenshot"/>
<img style = "text-align: center" src="assets/img1.png" alt="Figure 2: Graph 1"/>
<img style = "text-align: center" src="assets/img2.png" alt="Figure 3: Graph 2"/>
