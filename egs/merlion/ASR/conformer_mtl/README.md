## Introduction

This recipe trains a model for the Language IDentification (LID) task on a collection of Mandarin-English code-switched and monolingual data provided by the MERLion challenge organizer.

To run the recipe, one need to:
0. Download and configure [lhotse](https://lhotse.readthedocs.io/en/latest/getting-started.html), [k2](https://k2-fsa.github.io/k2/installation/index.html) and [icefall](https://icefall.readthedocs.io/en/latest/installation/index.html). Note that the installation of `k2` on clusters may require some additional configuration, see [Desh's guide to install k2](https://wiki.clsp.jhu.edu/index.php?title=Desh%27s_guide_to_install_k2). Also note that for `lhotse` installation, please use [this version](https://github.com/BorrisonXiao/lhotse-cxiao.git) as it contains some dependent additional data processing recipes.
1. Manually download the MERLion data proivded by the organizer;
2. Configure the pathes of the downloaded data in `./prepare.sh`;
3. Run `./prepare.sh` to pre-process the data with `lhotse`;
4. Run `./pretrain.sh` to pre-train the model with interpolated multi-task (LID + ASR) objectves on the monolingual data;
5. Run `./finetune.sh` to fine-tune the pre-trained model on the code-switched data in the target domain.