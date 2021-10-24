# Hybrid Linker
This repositry contains the source code adn datasets for our study titled as 
"<strong>Automated Recovery of Issue-Commit Links Leveraging Both Textual and Non-textual Data</strong>",
accepted at the <i>IEEE International Working Conference on Source Code Analysis and Manipulation (ICSME 2021)</i>.

An issue report documents the discussions around required changes in issue-tracking systems, 
while a commit contains the change itself in the version control systems.
Recovering links between issues and commits can facilitate many software evolution tasks such as bug localization, 
defect prediction, software quality measurement, and software documentation.
A previous study on over half a million issues from GitHub
reports only about 42.2\% of issues are manually linked by developers to their pertinent commits.
Automating the linking of commit-issue pairs can contribute to the improvement of the said tasks.
By far, current state-of-the-art approaches for automated commit-issue linking suffer from low precision, leading to unreliable results, 
sometimes to the point that imposes human supervision on the predicted links.
The low performance gets even more severe when there is a lack of textual information in either commits or issues.
Current approaches are also proven computationally expensive.

We propose Hybrid-Linker, an enhanced approach that overcomes such limitations by exploiting two information channels; 
(1) a non-textual-based component that operates on non-textual, automatically recorded information of the commit-issue pairs to predict a link, 
and (2) a textual-based one which does the same using textual information of the commit-issue pairs.
Then, combining the results from the two classifiers, Hybrid-Linker makes the final prediction.
Thus, every time one component falls short in predicting a link, the other component fills the gap and improves the results.
We evaluate Hybrid-Linker against competing approaches, namely FRLink and DeepLink on a dataset of 12 projects.
Hybrid-Linker achieves 90.1%, 87.8%, and 88.9% 
based on recall, precision, and F-measure, respectively. 
It also outperforms FRLink and DeepLink
by 31.3%, and 41.3%, regarding the F-measure.

You can use the data by accessing Zenodo link: https://doi.org/10.5281/zenodo.5067833

<iframe width="560" height="315" src="https://www.youtube.com/embed/WIIvoYicJ9k" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
