package mulan.classifier.lazy;

import java.util.HashSet;
import java.util.Set;

import weka.core.Instance;
import weka.core.Instances;

public class MultiLabelParticle {

	public int indexCentroid;

	public double cohesion;

	public double purity;

	public Set<Integer> categoryvector;

	public int numOfNeighbors;

	public int numLabels;

	public int[] labelIndices;

	public boolean weightByDistance = false;

	public double m_sigma = 2;

	public MultiLabelParticle(int indexCentroid, Instance instance,
			int labelIndices[], Instances kNN, double[] distances) {

		this.indexCentroid = indexCentroid;

		categoryvector = new HashSet<Integer>();

		for (int l = 0; l < labelIndices.length; l++) {

			if (instance.value(labelIndices[l]) == 1)
				categoryvector.add(l);
		}

		numOfNeighbors = kNN.size();

		this.numLabels = labelIndices.length;

		this.labelIndices = labelIndices;

		computeCohesion(instance, kNN, distances);

		computePurity(instance, kNN, distances);

	}

	public double getPurity() {

		return purity;
	}

	public double getCohesion() {

		return cohesion;
	}

	public void setCohesion(double cohesion) {
		this.cohesion = cohesion;
	}

	public double getForce(double distance) {

		double force = 0;

		force = Math.pow(getPurity(), getCohesion()) / Math.pow(distance, 2);

		if (Double.isInfinite(force) || Double.isNaN(force)) {

			System.out.println(getPurity());
			System.out.println(getCohesion());
			System.out.println(distance);

			System.err.println("Force equals to NaN or infinite");
			return -1000;
		}

		return force;
	}

	@Override
	public String toString() {
		return "index: " + indexCentroid;
	}

	public boolean equals(Object o) {

		Particle n = (Particle) o;

		return (indexCentroid == n.indexCentroid);
	}

	private void computeCohesion(Instance instance, Instances kNN,
			double[] euclideanDistances) {

		double[] m_weightsByRank = null;

		double sum = 0;

		if (weightByDistance) {

			// m_sigma = Math.ceil((/* 1/5 */0.20) * numOfNeighbors);

			m_sigma = 2;

			m_weightsByRank = new double[numOfNeighbors];

			for (int i = 0; i < numOfNeighbors; i++) {

				double value = Math.exp(-Math.pow(i / m_sigma, 2));

				m_weightsByRank[i] = value;

				sum += value;
			}

		}

		// Probabilities
		double PdifY = 0;
		double PdifX = 0;
		double PdifYdifX = 0;

		int index = 0;

		for (Instance k : kNN) {

			double dh = distanceHamming(instance, k);

			double de = euclideanDistances[index];

			PdifY += (dh / numOfNeighbors)
					* ((weightByDistance) ? (m_weightsByRank[index] / sum) : 1);

			PdifX += (de / numOfNeighbors)
					* ((weightByDistance) ? (m_weightsByRank[index] / sum) : 1);

			PdifYdifX += ((dh * de) / numOfNeighbors)
					* ((weightByDistance) ? (m_weightsByRank[index] / sum) : 1);

			index++;
		}

		// PdifX /= numOfNeighbors;

		// PdifY /= numOfNeighbors;

		// PdifYdifX /= numOfNeighbors;

		cohesion = 0.0;

		if (PdifY != 0) {

			cohesion += PdifYdifX * PdifX / PdifY;

			// cohesion+= PdifYdifX/ PdifY ;

		}

		if (PdifY != 1) {
			cohesion -= (1 - PdifYdifX) * PdifX / (1 - PdifY);
			// cohesion -= (PdifX - PdifYdifX) / (1 - PdifY);
		}
		// weights[f] = (NdYF[f] / NdY) - ((NdF[f] - NdYF[f]) /
		// (tuningSet.size() - NdY));

	}

	private double distanceHamming(Instance instance1, Instance instance2) {

		double dh = 0;

		for (int l = 0; l < numLabels; l++) {

			dh += (instance1.value(labelIndices[l]) == instance2
					.value(labelIndices[l])) ? 0.0 : 1.0;
		}

		return dh / numLabels;
	}

	private void computePurity(Instance instance, Instances kNN,
			double[] euclideanDistances) {

		purity = 0;

		int index = 0;

		for (Instance k : kNN) {

			double dh = distanceHamming(instance, k);

			double de = euclideanDistances[index++];

			purity += (1 - dh)/de;

		}

		 purity = 1 + (purity);
	}

	private void computePurity2(Instance instance, Instances kNN,
			double[] euclideanDistances) {

		purity = 0;

		int index = 0;

		double sum = 0;

		for (Instance k : kNN) {

			double dh = distanceHamming(instance, k);

			double de = euclideanDistances[index++];

			purity += Math.exp(-Math.pow((1 - dh) * de, 2));

			sum += Math.exp(-Math.pow(de, 2));
		}

		// Quiere decir que todos los vecinos son identicos a la particula, por
		// lo tanto la pureza es la maxima
		if (sum == 0)
			purity = 2;
		else
			purity = 1 + (purity / sum);

	}

}
