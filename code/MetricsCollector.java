
import java.util.*;

public class MetricsCollector {

    // Minimal Tuple class replacement
    public static class Tuple {
        private final String id;
        public Tuple(String id) { this.id = id; }
        public String getTupleId() { return id; }
    }

    private static final Map<String, Double> tupleLatencies = new HashMap<>();
    private static int missedTasks = 0;

    // Collects the latency of the control loop
    public static void recordLatency(Tuple tuple, double latency) {
        tupleLatencies.put(tuple.getTupleId(), latency);
    }

    // Tracks tasks that fail to meet deadlines
    public static void recordMissedTask(Tuple tuple) {
        missedTasks++;
    }

    // Monitors CPU and RAM usage (dummy version)
    public static void monitorResourceUtilization() {
        Runtime runtime = Runtime.getRuntime();
        long usedMemory = runtime.totalMemory() - runtime.freeMemory();
        System.out.println("CPU load monitoring not implemented (placeholder).");
        System.out.println("Used Memory: " + usedMemory / (1024 * 1024) + " MB");
    }
}
