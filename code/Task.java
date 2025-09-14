package project;

public class Task {
    private int id;
    private String type;
    private int dataSizeBytes;
    private int cpuReqMI;
    private int deadlineMs;
    private int urgency;
    private double energyEst;
    private int sourceId;
    private int arrivalMs;

    public Task(int id, String type, int dataSizeBytes, int cpuReqMI,
                int deadlineMs, int urgency, double energyEst,
                int sourceId, int arrivalMs) {
        this.id = id;
        this.type = type;
        this.dataSizeBytes = dataSizeBytes;
        this.cpuReqMI = cpuReqMI;
        this.deadlineMs = deadlineMs;
        this.urgency = urgency;
        this.energyEst = energyEst;
        this.sourceId = sourceId;
        this.arrivalMs = arrivalMs;
    }

    // 🔹 Add getter methods
    public int getId() { return id; }
    public String getType() { return type; }
    public int getDataSizeBytes() { return dataSizeBytes; }
    public int getCpuReqMI() { return cpuReqMI; }
    public int getDeadlineMs() { return deadlineMs; }
    public int getUrgency() { return urgency; }
    public double getEnergyEst() { return energyEst; }
    public int getSourceId() { return sourceId; }
    public int getArrivalMs() { return arrivalMs; }

    // 🔹 Urgency check
    public boolean isUrgent() {
        return urgency >= 8 || deadlineMs < 200; // threshold can be tuned
    }

    @Override
    public String toString() {
        return "Task{id=" + id +
                ", type='" + type + '\'' +
                ", size=" + dataSizeBytes + " bytes" +
                ", cpu=" + cpuReqMI + " MI" +
                ", deadline=" + deadlineMs + " ms" +
                ", urgency=" + urgency +
                ", energy=" + energyEst +
                ", source=" + sourceId +
                ", arrival=" + arrivalMs + " ms}";
    }
}
