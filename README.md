Code to generate the results presented in "Preseving physically important variables in optimal event selections: A case study in Higgs physics"

A full run consists of:
* training the model
* applying the model for inference
* evaluating its performance

To set up the environment (on a CentOS machine):
```
source bin/activate
source setup_env.sh
```

To run a training campaign:
```
source bin/activate
python2 RunTrainingCampaign.py --confpath ~/datasmall/Hbb_adv_test/MINEAdversaryMaster.conf --nrep 5
```

To run an evaluation campaign:
```
source bin/activate
python2 RunEvaluationCampaign.py ~/datasmall/Hbb_adv_test/MINEAdversaryMaster_slice_0.*
```

To produce the summary plots:
```
source bin/activate
python MakeGlobalPlots.py --plotdir ~/datasmall/Hbb_adv_test/ ~/datasmall/Hbb_adv_test/MINEAdversaryMaster_slice_0.* 
```