package mulan.classifier.transformation;

import mulan.classifier.OutputImpreciseLabelRanking;
import mulan.data.MultiLabelInstances;
import weka.classifiers.Classifier;
import weka.core.Instance;

public class ImpreciseCalibratedLabelRanking {
	 /**
     * Binary imprecise models
     */
    private ImpreciseBinaryRelevance one_VS_rest_models;
    
    /**
     * Pairwise imprecise models
     */
    private ImprecisePairwise one_VS_one_models;
    
	/**
    * whether to consider predicted probability intervals or binary outputs
    */
   protected boolean probability_intervals_output;
   
   /**
    * Constructor that initializes the learner 
    * By default, output of pairwise learners are probability intervals
    *
    */
   public ImpreciseCalibratedLabelRanking() {
       probability_intervals_output = true;
       one_VS_rest_models = new ImpreciseBinaryRelevance();
       one_VS_one_models = new ImprecisePairwise();
   }
   
   /**
    * Constructor that initializes the learner
    * @param intervals_output whether the output 
    * is in form of probability intervals
    */
   
   public ImpreciseCalibratedLabelRanking(boolean intervals_output) {
       probability_intervals_output = intervals_output;
       one_VS_rest_models = new ImpreciseBinaryRelevance();
       one_VS_one_models = new ImprecisePairwise(intervals_output);
   }
   
   /**
    * Sets whether the prediction is binary 
    * or in the form of probability intervals
    * @param probability_intervals true if the prediction is 
    * in the form of probability intervals, 0 otherwise
    */
   
   public void setFormOutput(boolean probability_intervals) {
	   probability_intervals_output = probability_intervals;
   }
   
   /**
    * Returns whether the predictions of the imprecise binary 
    * classifiers are in form of probability intervals
    * @return boolean value which indicates the mentioned condition
    */
   
   public boolean getFormOutput() {
	   return probability_intervals_output;
   }
   
   /**
    * Builds the imprecise Calibrated Label Ranking 
    * Classifier from the training set
    * Builds the one VS one imprecise classifiers X
    * and the one VS all imprecise classifier
    * @param trainingSet the training instances
    * @throws Exception if there is a problem in the building
    */
   
   public void build(MultiLabelInstances trainingSet) throws Exception {
	   one_VS_rest_models.buildInternal(trainingSet);
	   one_VS_one_models.buildInternal(trainingSet);
   }
   
   /**
    * Does a prediction for an unlabeled instance
    * It takes into account whether the output of the pairwise 
    * classifiers is of the form of scores or probability intervals
    * @param instance the unlabeled instance, passed 
    * through the pairwise transformation filter
    * @return the prediction made for the instance
    */
   
   public OutputImpreciseLabelRanking makePredictionInternal(Instance instance) {
	   int num_labels = one_VS_one_models.getNumLabels();
	   OutputImpreciseLabelRanking output = new OutputImpreciseLabelRanking(num_labels);
	   double[] lower_probabilities, upper_probabilities;
	   double lower_probability_virtual, upper_probability_virtual;
	   double partial_lower_virtual, partial_upper_virtual;
	   double[] lower_probabilities_BR, upper_probabilities_BR;
	   double[] lower_scores, upper_scores;
	   double lower_score_virtual, upper_score_virtual;
	   
	   upper_scores = new double[num_labels];
	   upper_probabilities = new double[num_labels];
	   upper_probabilities_BR = new double[num_labels];

	   lower_probabilities_BR = one_VS_rest_models.computeLowerUpperProbabilities(instance, upper_probabilities_BR);
	   
	   if(probability_intervals_output) {
		   // Sets the partial ranking from probability intervals
		   lower_probabilities = one_VS_one_models.computeLowerUpperProbabilities(instance, upper_probabilities);
	   
		   /* Compute the probability interval for the virtual label
		    * Update the lower (upper) probabilities for each label 
		    * considering the lower (upper) probabilities for the virtual label
		    */
		   lower_probability_virtual = 0;
		   upper_probability_virtual = 0;
		   
		   for(int i = 0; i < num_labels; i++) {
			   lower_probabilities[i]+=lower_probabilities_BR[i];
			   upper_probabilities[i]+=upper_probabilities_BR[i];
			   partial_lower_virtual = 1 - upper_probabilities_BR[i];
			   partial_upper_virtual = 1 - lower_probabilities_BR[i];
			   lower_probability_virtual = lower_probability_virtual + partial_lower_virtual; 
			   upper_probability_virtual = upper_probability_virtual + partial_upper_virtual; 
		   }
		   
	   }
	   
	   else {
		   // Compute ranking from scores
		   lower_scores = one_VS_one_models.getLowerUpperScores(instance, upper_scores);
	   
		   /* Update the lower and upper scores for each label and compute the 
		    * lower and upper scores of the virtual label 
		    */
		   lower_score_virtual = 0;
		   upper_score_virtual = 0;
		   
		   for(int i = 0; i < num_labels; i++) {
			   if(lower_probabilities_BR[i] > 0.5) {
				   lower_scores[i]+=1;
				   upper_scores[i]+=1;
			   }
			   
			   else if(upper_probabilities_BR[i] < 0.5) {
				   lower_score_virtual+=1;
				   upper_score_virtual+=1;
			   }
			   
			   else {
				   upper_scores[i]+=1.0;
				   upper_score_virtual+=1.0;
			   }
		   }
		   lower_probabilities = one_VS_one_models.getLowerUpperProbabilitiesFromScores(lower_scores, upper_scores, upper_probabilities);
		   
		   lower_probability_virtual = lower_score_virtual/num_labels;
		   upper_probability_virtual = upper_score_virtual/num_labels;

	   }
	   
	   // Compute the partial predictions
	   output.computePartialRankingFromProbabilities(lower_probabilities, upper_probabilities);
	   output.computePartialPredictions(lower_probabilities, upper_probabilities,lower_probability_virtual, upper_probability_virtual);
	   
	   return output;
   }
   
    
}
