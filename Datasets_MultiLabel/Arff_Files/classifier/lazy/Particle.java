package mulan.classifier.lazy;

import java.util.ArrayList;
import java.util.Comparator;

public class Particle implements Comparable<Particle>, Comparator<Particle> {

	public int indexCentroid;

	public double distance;

	public int numberPerClass;

	public double cohesion;

	public double purity;

	public int classIndex;

	public Particle() {
	}

	public Particle(int indexCentroid, int classIndex, double distance) {

		super();

		this.indexCentroid = indexCentroid;

		this.classIndex = classIndex;

		this.distance = distance;

		numberPerClass = 0;

		cohesion = 0;

		purity = 1;

	}

	public Particle(int index, int classValue, ArrayList<Particle> knn,
			double probabilitiesPerClass[], int numOfNeighbors) {

		this.indexCentroid = index;

		this.distance = 0;

		this.classIndex = classValue;

		cohesion = 0;

		numberPerClass = 0;

		int numberClasses = probabilitiesPerClass.length;

		double[] distances = new double[numberClasses];

		double[] numOfNeighborsPerClass = new double[numberClasses];

		double distancek = 0;
		double distanceT = 0;

		for (int k = 0; k < knn.size(); k++) {

			Particle p = knn.get(k);

			if (p.getClassOfParticle() == classIndex) {
				numberPerClass++;
				distancek += p.getDistance();
			}

			distanceT += p.getDistance();

			distances[p.getClassOfParticle()] += Math.pow(p.getDistance(), 2);
			numOfNeighborsPerClass[p.getClassOfParticle()]++;
		}

		// si no existen vecinos mas cercanos de mi clase entre los k primeros
		// vecinos entonces se
		// convierte en el caso base 1/r*r
		if (numberPerClass == 0) {
			cohesion = 1;
			purity = 1;

		} else {

			for (int c = 0; c < numberClasses; c++) {

				if (distances[c] != 0) {

					distances[c] /= numOfNeighborsPerClass[c];

					if (classIndex == c)
						cohesion -= distances[c];
					else {
						cohesion += (probabilitiesPerClass[c] / (1 - probabilitiesPerClass[classIndex]))
								* distances[c];
					}
				}
			}

			if (Double.isInfinite(cohesion) || Double.isNaN(cohesion)) {
				System.err.println("Cohesion equals to NaN or infinite");
				System.exit(1);
			}

			purity = 1 + ((double) distancek / distanceT);
		}

		if (Double.isInfinite(cohesion) || Double.isNaN(cohesion)) {
			System.err.println("Cohesion equals to NaN or infinite");
			System.exit(1);
		}

		if (Double.isInfinite(purity) || Double.isNaN(purity)) {
			System.err.println("Purity equals to NaN or infinite");
			System.exit(1);
		}

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

	public int getClassOfParticle() {

		return classIndex;

	}

	public double getDistance() {
		return distance;
	}

	public void setDistance(double distance) {
		this.distance = distance;
	}

	public double getForce() {
		double force = 0;

		force = Math.pow(getPurity(), getCohesion())
				/ Math.pow(getDistance(), 2);

		if (Double.isInfinite(force) || Double.isNaN(force)) {

			System.out.println(getPurity());
			System.out.println(getCohesion());
			System.out.println(getDistance());

			System.err.println("Force equals to NaN or infinite");
			return -1000;
		}

		return force;
	}

	public double getSimpleForce() {

		double force = 0;

		force = 1 / Math.pow(getDistance(), 2);

		if (Double.isInfinite(force) || Double.isNaN(force)) {
			System.err.println("Force equals to NaN or infinite");
			System.exit(1);
		}

		return force;
	}

	@Override
	public String toString() {
		return "distance: " + distance;
	}

	public boolean equals(Object o) {

		Particle n = (Particle) o;

		return (indexCentroid == n.indexCentroid && distance == n.distance);
	}

	@Override
	public int compare(Particle o1, Particle o2) {

		return o1.compareTo(o2);
	}

	@Override
	public int compareTo(Particle o) {

		if (distance < o.distance)
			return -1;
		else {
			if (distance == o.distance) {
				if (numberPerClass > o.numberPerClass)
					return -1;
				else
					return 1;
			} else
				return 1;
		}

	}
}

