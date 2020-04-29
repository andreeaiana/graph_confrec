# User Interface

A prototype of GraphConfRec, containing some of the best-performing models, is temporarily running at: http://westpoort.informatik.uni-mannheim.de/

To run the user interface:

1. Download the necessary data from https://drive.google.com/drive/folders/1fTfA98OBKk04snqHnU7PCsvNN18Yzs4Q?usp=sharing into `./data/`.
2. Change the current port to the desired address. 
3. ```python Server.py```

The user feedback is saved in `./db/feedback.db`. Run ```python process_feedback.py``` to process the feedback per model and save it as _csv_ and _pkl_ files.
