# Spoken "AAC-Like" corpora 

- We have dailyDialog and BNC2014 corpora extracted and tweaked
- Main tweaks - removal of double words like yeah and removal of speech components like 'mm' 
- Removal of quotes and other common written only aspects
- For the purposes of our tool we then add in typos and remove the spaces


- We then train a t5-small model on both these. Note we did it in two stages. First the dailyDialog (it has a lot of quotes in it). and next the BNC corpora



