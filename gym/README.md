# CSC 587: Genetic Pacman

The goal of this research is to compare different algorithms and models to play the game of packman optimally.


## Notes

### ALE
https://ale.farama.org/environments/pacman/ -> For the api interface into the pacman game

#### Objective
So the obejctive space can be the image or it can be the ram. For neat I'm not quite sure what I want it to be. It might be fun to train a neat model to detect the positions of the ghosts and of pacman and the pellets. And then feed that into a downstream one.

### GYM
https://gymnasium.farama.org/api/env/ -> For the gym api that is used by ale
