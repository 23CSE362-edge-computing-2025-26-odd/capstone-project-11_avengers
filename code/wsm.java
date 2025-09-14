import java.util.Map;

public class WSM {
    // Weights for each criterion
    private static final double W_LATENCY = 0.5;
    private static final double W_PRIORITY = 0.5;

    public static double calculateScore(FogDeviceInfo device) {
        // Higher priority is good, lower latency is good
        double latencyScore = 1.0 / (1.0 + device.getLatency()); // normalize
        double priorityScore = device.getPriority();

        // Weighted sum
        return (W_LATENCY * latencyScore) + (W_PRIORITY * priorityScore);
    }
}
