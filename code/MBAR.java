package project;

import java.util.List;

public class MBAR {
    public static double calculatePriority(Task task, List<FogDeviceInfo> fogDevices) {
        // Reliability factor: pick highest available fog reliability
        double bestReliability = fogDevices.stream()
                .mapToDouble(FogDeviceInfo::getReliability)
                .max().orElse(0.5);

        // Normalize factors
        double normUrgency = task.getUrgency() / 10.0;
        double normDeadline = 1.0 - (Math.min(task.getDeadlineMs(), 1000) / 1000.0);

        // Weights (deadline 0.3, urgency 0.4, reliability 0.3)
        double w1 = 0.3;
        double w2 = 0.4;
        double w3 = 0.3;

        double score = (w1 * normDeadline) + (w2 * normUrgency) + (w3 * bestReliability);
        return Math.max(0.0, Math.min(1.0, score));
    }
}
