What it is
Eyevoice connects to a camera that is streamed over a server. It relies on tensorflow Machine Learning model to be able to recognize what's happening around the user at any time and send him/her audio hints with updates about their surroundings. For now, it detects people in the scene and tells the user how many people they are talking to.

How we built it
We used a pretrained single shot multibox detector (MobileNet, https://arxiv.org/abs/1704.04861) to be able to find and reveal information about the user surroundings. The input comes from a camera (embedded in glasses/phones) and is transmitted via our app to be analyzed and then inferred on. Then we use voice capabilities widely available on phones to transmit audio hints to the user.

Challenges we ran into
The main challenge was trying to build the entire pipeline with glasses at first, we pivoted to using a phone. Another challenge was also in objects/people recognition.

Accomplishments that we're proud of
We were able to take live video and count how many people there are on every frame
We were able to accomplish the basic use case of audio feedback about the user surroundings
What we learned
What's next for EyeVoice
It would be interesting if we could build use glasses with embedded cameras that are connected to our server instead of a phone. That would be more convenient. Also, we could add other options like description of the scene around the user, a recognition of known faces among everyone around the user where known faces are added to the user's database by saying "Nice to meet you [person's name]", an option of calling automatically their favorite number if they get lost (which tends to happen quite a lot) etc.

This tool was built with python, javascript, react, google-cloud, tensorflow, opencv
