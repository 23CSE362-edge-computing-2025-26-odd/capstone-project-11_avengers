public class FogDeviceInfo {
    private String name;
    private double latency;
    private double priority;

    public FogDeviceInfo(String name, double latency, double priority) {
        this.name = name;
        this.latency = latency;
        this.priority = priority;
    }

    public String getName() {
        return name;
    }

    public double getLatency() {
        return latency;
    }

    public double getPriority() {
        return priority;
    }

    public String toString() {
        return "FogDevice{name=" + name +
               ", latency=" + latency +
               ", priority=" + priority + "}";
    }
}
