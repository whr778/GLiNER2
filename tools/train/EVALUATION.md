tools/train/train.py

1. Create a summary like what is displayed at the end of each epoch that is displayed after the detailed blind-test at the end of training.

2. While reading the training files keep a 3-character iso 639-2 language code for each sample.  I need to keep this list in-sync with the actual training data, it needs to survive shuffles.  If you need to run Language ID aka LID use https://github.com/whr778/lumi_language_id use my fork, not the original, the original is no longer supported, I have updated this fork.  It is published on pypi as https://pypi.org/project/lumi-language-id-2/. To get the 3-digit language code from the 2-digit LID, use the python langcodes package.
 
3. Create a yaml configuration option eval_by_language, default is False.
 
4. If the yaml configuration eval_by_language is True, perform a detailed evaluation and summary from step 1 for each language in the dataset in alphabetical order.  After the detailed by language evaluation is completed do a detailed evaluation of all of the data, as we do now.  Logging should be detailed... E.g. processing laguage XX 
 
5. Please create a plan and document it in tools/train/EVALUATION_PLAN.md
