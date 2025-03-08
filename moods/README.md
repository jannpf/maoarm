# Moods

This python package provides the mood functionality, including random mood drift, character definition and gesture impact.
1. `cat.py`
    * Provides the class 'Cat'
    * Implements the Metropolis Hastings Algorithm for mood drift (see details below)
    * Allows the definition of a Cat Character profile by providing distribution parameters
    * Includes plotting with contour lines of the distributions and arrows indicating manual offsets (as utilized by gesture impacts in the arm module)
2. `simulate.py`
    * simulate the mood drift for a given profile and visualize it
    * while running, it accepts mood offsets as cmdline parameters to simulate gestures
    * overrides need to be passed as tuples, e.g. (-0.5, 0.5) for offsetting the current mood with -0.5 valence and +0.5 arousal

## Mood drift

- Emotional model (state is defined by valence and arousal, hence the mood state can be modeled as point in this two-dimensional space)
- These two variables drift randomly across this mood space
- The mood drift should not be arbitrary, since too high fluctuations are undesired, but rather low frequency drifts, where the current mood also depends on previous states
- Markov Chains are well suited for modeling this dependent behaviour + Monte-Carlo for randomness -> MCMC
- For the natural mood drift, the [Metropolis-Hastings-Algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) is employed to gradually traverse randomly between mood states.
- Instead of using this MCMC method to approximate the distribution, each sample is recorded as new mood state
- 4 main configuration elements:
    - proposal distribution sigma (basically determines how big each change in mood will generally be, read it as temper. The higher the standard deviation of the proposal distribution, the higher the likelihood of more drastic mood changes)
    - the underlying distribution profile. Currently implemented as Mixed - Multivariate Gaussian. For defining a cat character profile, you can define an arbitrary number of gaussians defined by mu1 and sigma1 (valence dimension), mu2 and sigma2 (arousal dimension)
    - Covariance factor rho: determines the covariance between the two factors valence and arousal
    - weight: increase the weight of certain gaussians to increase the 'pull factor' of that distribution.
- The resulting, combined probability distribution is traversed with MH. This allows the definition of arbitrary complex character profiles, determining the probability of the cat being in certain state range, how likely it is to traverse to a specific other state. One example can be found in cat_characters/spot.json.
- Here 4 gaussians are defined, each representing a quadrant in the valence-arousal model between -1 and 1. The covariance is chosen, so that the cat's mood is likely to drift towards the center rather than to a neighboring quadrant, so that the transition to any other state is equally likely. (If the covariance would be kept at 1, the likelihood to traversing to adjacent quadrants would be higher, while moving diagonally to the opposite quadrant is less.)
- sigma1, 2 and 3 contours are displayed in the plots, so the distributions can be inspected and tuned.
- for each cat character, the impact of gestures can also be defined as key value pairs in the json file.