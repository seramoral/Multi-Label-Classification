package mulan.classifier.lazy;

import java.util.ArrayList;
import java.util.Random;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.lazy.MultiLabelKNN;
import mulan.data.MultiLabelInstances;

public class ML_DGC2 extends MultiLabelKNN {

	private static final long serialVersionUID = 1657785045775363618L;

	// It stores the particles. Each particle is a single particle but the
	// purity and the cohesion are computed.
	private ArrayList<MultiLabelParticle> particles;

	public ML_DGC2(Integer numOfNeighbors) {

		super(numOfNeighbors);

	}

	@Override
	protected void buildInternal(MultiLabelInstances train) throws Exception {

		super.buildInternal(train);

		// label attributes don't influence distance estimation
		String labelIndicesString = "";
		for (int i = 0; i < numLabels - 1; i++) {
			labelIndicesString += (labelIndices[i] + 1) + ",";
		}

		labelIndicesString += (labelIndices[numLabels - 1] + 1);
		dfunc.setAttributeIndices(labelIndicesString);
		dfunc.setInvertSelection(true);

		lnn = new NearestNeighborsPerClass();
		lnn.setDistanceFunction(dfunc);
		lnn.setInstances(this.train);
		lnn.setMeasurePerformance(false);
		lnn.setSkipIdentical(true);

		// Create the data particles
		createDataParticles();
	}

	private void createDataParticles() {

		try {

			// Se crea un arreglo de longitud igual a a cantidad de instancias
			// en el training set
			particles = new ArrayList<MultiLabelParticle>(train.numInstances());

			double minCohesion = Double.MAX_VALUE;

			double maxCohesion = Double.MIN_VALUE;

			// For each instance compute its purity and cohesion
			for (int i = 0; i < train.numInstances(); i++) {

				Instances kNN = lnn.kNearestNeighbours(train.instance(i),
						numOfNeighbors);

				MultiLabelParticle part = new MultiLabelParticle(i, train.instance(i), labelIndices, kNN, lnn.getDistances());

				particles.add(part);

				// To normalize the cohesions
				if (minCohesion > part.cohesion)
					minCohesion = part.cohesion;

				if (maxCohesion < part.cohesion)
					maxCohesion = part.cohesion;

			}

			// Normalizar las cohesiones

			for (MultiLabelParticle p : particles) {

				double cohesionT = (p.cohesion - minCohesion)
						/ (maxCohesion - minCohesion);

				// quiere decir que maxCohesion es igual a minCohesion
				if (Double.isInfinite(cohesionT) || Double.isNaN(cohesionT)) {
					cohesionT = 1;
				}

				p.cohesion = cohesionT;

			}

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	@Override
	public TechnicalInformation getTechnicalInformation() {

		TechnicalInformation result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR, "O. Reyes");
		result.setValue(Field.TITLE,
				"Multi-label Data Gravitation Classification algorithm");
		result.setValue(Field.JOURNAL, "-");
		result.setValue(Field.VOLUME, "-");
		result.setValue(Field.YEAR, "-");
		result.setValue(Field.PAGES, "-");
		result.setValue(Field.PUBLISHER, "-");

		return result;
	}

	@Override
	protected MultiLabelOutput makePredictionInternal(Instance test)
			throws Exception, InvalidDataException {

		double[] confidences = new double[numLabels];
		boolean[] predictions = new boolean[numLabels];

		int[] indiceskNN = ((NearestNeighborsPerClass) lnn)
				.kNearestNeighboursIndices(test, numOfNeighbors);

		double distance[] = lnn.getDistances();

		double[] positivegravitationalForce = new double[numLabels];

		double[] negativeGravitationalForce = new double[numLabels];

		// For each neighbors
		int i = 0;

		for (int index : indiceskNN) {

			MultiLabelParticle p = particles.get(index);

			double currentDistance = distance[i++];

			// The instances are identical, the test instance is classified as
			// the training instance
			if (currentDistance == 0) {

				for (int l = 0; l < numLabels; l++) {

					if (p.categoryvector.contains(l)) {
						predictions[l] = true;
						confidences[l] = 1.0;
					} else {
						predictions[l] = false;
						confidences[l] = 0.0;
					}

				}

				MultiLabelOutput mlo = new MultiLabelOutput(predictions,
						confidences);
				return mlo;
			}

			// For each relevant label
			for (int l = 0; l < numLabels; l++) {

				if (p.categoryvector.contains(l))
					positivegravitationalForce[l] += p.getForce(currentDistance);
				else
					negativeGravitationalForce[l] += p.getForce(currentDistance);
			}

		}

		for (int l = 0; l < numLabels; l++) {

			if (positivegravitationalForce[l] > negativeGravitationalForce[l])
				predictions[l] = true;

			if (positivegravitationalForce[l] < negativeGravitationalForce[l])
				predictions[l] = false;

			if (positivegravitationalForce[l] == negativeGravitationalForce[l]) {
				Random rnd = new Random();
				predictions[l] = (rnd.nextInt(2) == 1) ? true : false;
			}

			// ranking function
			if ((positivegravitationalForce[l] + negativeGravitationalForce[l]) == 0) {
				confidences[l] = positivegravitationalForce[l];
			} else {
				confidences[l] = positivegravitationalForce[l]
						/ (positivegravitationalForce[l] + negativeGravitationalForce[l]);
			}

		}

		MultiLabelOutput mlo = new MultiLabelOutput(predictions, confidences);
		return mlo;

	}
}
