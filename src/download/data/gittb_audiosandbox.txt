<h1>Outline of steps for attempting NLP with audio instead of text. *Everything is WIP*</h1>

<h2>The thought process behind the project:</h2>

    - Tone and pitch changes inter-sentence determine importance in words far better than end markings (?, !, ., etc).

    - If the goal of NLP is to model the language as a whole, then we are inadvertently modeling the thoughts and state
        of mind behind the words. If spoken language is derived from thought, then written language would be derived
        from spoken language. Using the more rich data in audible language may help the model may help understand the
        thought.

    - TL;DR: Spoken language might be a closer example to modeling meaning behind words than text.


<h2>Possible downsides:</h2>

    - Without lemitizing and vectorizing spoken words, the model could have a much harder time understanding waveforms
        instead of vectors.

    - The processing power required.


<h2>Breaking down the process:</h2>

<h3>- (Preprocessing)</h3> Text embedding to audio challenges (Goal: transformers are fed full sentences, attempt to do the same.
    with audio) (Might forgo this step altogether.)

    * General audio cleaning and loading:
        
        - Load .wav .mp3 .flv .ogg to numpy arrays
        
        - Interpolate existing bitrate into desired bitrate to be uniform.
        
        - Throw out low hz audio.

        - Normalize audio between 1 and -1 without silencing the entire clip due to a mic bump or loud sound. (clipping peaks)
        
        - format audio from sound pressure level array into short sample spectrograms with overlapping windows for recreation.

<h3>- (Main model)</h3> Base on proven transformer architecture (specifically GPT/GPT-2). 
<br>GPT-2 Sample Code: https://github.com/openai/gpt-2
<br>Attention Is All You Need Paper link: https://arxiv.org/abs/1706.03762


    * Training (unsupervised)

        - Feed model similar to how entire phrases are fed to GPT, an array of the waveform (of varying length) and
            an array the positional data for that waveform

        - The answer is next N seconds of audio data following the waveform (also varying)

<h3>- Data</h3>

    GOAL: Diverse and large dataset, mainly containing conversational data. Both formal and informal.

    * Needs to contain wide varieties of emotion, emphasis, purpose (persuasive, questioning, debating, casual)

    * Diverse amounts of accents, dialect and possible lanugages.

    * Possible sources:

        - Podcasts
        - youtube vlogs
        - infomational videos
        - news and talkshows
        - college lectures
        - audio books
        - Poetry? *could give good data on expression reading


<h2>The Plan:</h2>

 1. Train preprocessing model
 
 2. Use model further clean and normalize dataset

 2. Cultivate dataset

 3. Train progressively larger models using the outlined training process.
