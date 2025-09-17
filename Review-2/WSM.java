package project;

public class WSM {
    public static double calculatePriority(Task task) {
        // Normalize factors (assumes max deadline=1000, urgency=10, energy=5 for scaling)
        double normDeadline = 1.0 - (Math.min(task.getDeadlineMs(), 1000) / 1000.0);
        double normUrgency = task.getUrgency() / 10.0;
        double normEnergy = Math.min(task.getEnergyEst(), 5.0) / 5.0;

        // Weights (tune as per case study)
        double w1 = 0.4; // deadline
        double w2 = 0.4; // urgency
        double w3 = 0.2; // energy

        double score = (w1 * normDeadline) + (w2 * normUrgency) + (w3 * normEnergy);
        return Math.max(0.0, Math.min(1.0, score)); // clamp to [0,1]
    }
}
