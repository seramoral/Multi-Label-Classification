package mulan.classifier.lazy;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class BRCredalkNN extends BRkNN{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public BRCredalkNN(int numOfNeighbors, ExtensionType ext) {
		super(numOfNeighbors, ext);
	}
	
	 /**
     * Calculates the confidences of the labels, based on the neighboring
     * instances
     *
     * @param neighbours
     *            the list of nearest neighboring instances
     * @param distances
     *            the distances of the neighbors
     * @return the confidences of the labels
     */
	@Override
	protected double[] getConfidences(Instances neighbours, double[] distances) {
        double[] confidences = new double[numLabels];
        Instance neighbor;
        double neighbor_labels = 0;
        int num_neighbours = neighbours.numInstances();
        double threshold = (double)num_neighbours/2;

        for (int i = 0; i < num_neighbours; i++) {
            // Collect class counts
            neighbor = neighbours.instance(i);
            distances[i] = distances[i] * distances[i];
            distances[i] = Math.sqrt(distances[i] / (train.numAttributes() - numLabels));

            for (int j = 0; j < numLabels; j++) {
                double value = Double.parseDouble(neighbor.attribute(
                        labelIndices[j]).value(
                        (int) neighbor.value(labelIndices[j])));
                if (Utils.eq(value, 1.0)) {
                    confidences[j] += 1.0;
                    neighbor_labels+=1.0;
                }
            }
        }
        
        avgPredictedLabels = (int) Math.round(neighbor_labels / num_neighbours);
        
        for(int j = 0; j < numLabels; j++) {
        	if(confidences[j] > threshold)
        		confidences[j] = (confidences[j]-1)/num_neighbours;
        	
        	else if(confidences[j] < threshold)
        		confidences[j] = (confidences[j]+1)/num_neighbours;
        	
        	else
        		confidences[j] = confidences[j]/num_neighbours;
        }
        
        
        return confidences;
    }

    /**
     * used for BRknn-a
     *
     * @param confidences the probabilities for each label
     * @return a bipartition
     */

}
