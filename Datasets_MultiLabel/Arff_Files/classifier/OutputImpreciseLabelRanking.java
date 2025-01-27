package mulan.classifier;

public class OutputImpreciseLabelRanking {
	/**
	 * Matrix of partial orders. Dimension = num_labels x num_labels 
	 * m_{ij} = true if y_i dominates y_j, false otherwise. 
	 */
	boolean[][] partial_orders;
	
	/**
	 * Array of imprecise predictions. 
	 * 0 if the i-th label predicted as irrelevant
	 * 1 if the i-th labels is predicted as relevant
	 * 0'5 if it is not determined the relevance of y_i
	 */
	
	double[] imprecise_predictions;
	
	/**
	 * Constructor for initialization given the number of labels
	 * @param num_labels the number of labels
	 */
	
	public OutputImpreciseLabelRanking(int num_labels){
		partial_orders = new boolean[num_labels][num_labels];
		imprecise_predictions = new double[num_labels];
	}
	
	/**
	 * Constructor for initialization given the matrix of partial orders
	 * @param orders the matrix of partial orders
	 */
	
	public OutputImpreciseLabelRanking(boolean[][] orders) {
		setOrders(orders);
	}
	/**
	 * Constructor for initialization given the matrix of partial orders
	 * and the array of imprecise predictions
	 * @param orders the matrix of partial orders
	 * @param predictions the imprecise predictions
	 */
	
	public OutputImpreciseLabelRanking(boolean[][] orders, double[] predictions) {
		setOrders(orders);
		setPartialPredictions(predictions);
	}
	
	/**
	 * Sets the matrix of partial orders
	 * @param orders the matrix of partial orders
	 */
	
	public void setOrders(boolean[][] orders) {
		int num_labels = orders.length;
		
		for(int i = 0; i < num_labels; i++) {
			partial_orders[i][i] = orders[i][i];		
			
			for(int j = i + 1; j < num_labels; j++) {
				partial_orders[i][j] = orders[i][j];
				partial_orders[j][i] = orders[j][i];
			}
		}
	}
	/**
	 * Sets the partial predictions
	 * @param predictions the partial predictions
	 */
	
	public void setPartialPredictions(double[] predictions) {
		int num_labels = predictions.length;
		
		for(int i = 0; i < num_labels; i++)
			imprecise_predictions[i] = predictions[i];
	}
	
	/**
	 * Gets the matrix of partial orders
	 * @return the matrix of partial orders
	 */
	
	public boolean[][] getPartialOrders(){
		return partial_orders;
	}
	
	/**
	 * 
	 * @return the array with the imprecise predictions
	 */
	
	public double[] getImprecisePredictions() {
		return imprecise_predictions;
	}
	
	/**
	 * Computes the partial ranking given the lower and upper probabilities
	 * a label y_i dominates another label y_j iif 
	 * the lower probability of y_i > the upper probability of y_j
	 * @param lower_probabilities the lower probabilities of each label
	 * @param upper_probabilities the upper probabilities of each label
	 */
	
	public void computePartialRankingFromProbabilities(double[] lower_probabilities, double[] upper_probabilities) {
		int num_labels = lower_probabilities.length;
		
		partial_orders = new boolean[num_labels][num_labels];
		
		for(int i = 0; i < num_labels; i++) {
			partial_orders[i][i] = false;
			
			for(int j = i+1; j < num_labels; j++) {
				partial_orders[i][j] = lower_probabilities[i] > upper_probabilities[j];
				partial_orders[j][i] = lower_probabilities[j] > upper_probabilities[i];
			}
		}
	}
	
	/**
	 * Computes the partial predictions from the probability interval
	 * of each label and the probability interval of the virtual label 
	 * For each label, y_j is predicted as relevant if, and only if, 
	 * underline{P}(y_j) > overline{P}(y_0)
	 * y_j is predicted as irrelevant iif underline{P}(y_0) > overline{P}(y_j)
	 * Otherwise, y_j is predicted as undetermined. 
	 * @param lower_probabilities the lower probabilities of the class labels
	 * @param upper_probabilities the upper probabilities of the class labels
	 * @param lower_probability_virtual the lower probability of the virtual label
	 * @param upper_probability_virtual the upper probability of the virtual label
	 */
	
	public void computePartialPredictions(double[] lower_probabilities, double[] upper_probabilities, double lower_probability_virtual, double upper_probability_virtual) {
		int num_labels = lower_probabilities.length;
		
		for(int i = 0; i < num_labels; i++) {
			if(lower_probabilities[i] > upper_probability_virtual)
				imprecise_predictions[i] = 1;
			
			else if(upper_probabilities[i] < lower_probability_virtual)
				imprecise_predictions[i] = 0;
			
			else
				imprecise_predictions[i] = 0.5;
		}
	}
	
}
