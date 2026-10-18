"""Synthetic evaluation corpus: fictional Acme Robotics products and policies.

All documents are fictional and generated deterministically from a fixed seed.
In production, replace with your own document collection.
"""

from dataclasses import dataclass

# Deterministic vocabulary blocks — each doc is assembled from these so that
# the corpus is fully reproducible without any RNG at import time.
_TOPICS = [
    # 0: warranty policy
    ("Warranty Policy", "warranty",
     "Acme Robotics warranty coverage extends for twenty four months from the date of purchase. "
     "The warranty covers manufacturing defects in materials and workmanship. "
     "It does not cover damage from misuse, unauthorized modification, or normal wear. "
     "To file a warranty claim, contact support with the serial number and proof of purchase. "
     "Approved claims are repaired or replaced within ten business days."),
    # 1: return policy
    ("Return Policy", "returns",
     "Products may be returned within thirty days of delivery for a full refund. "
     "Items must be unused and in original packaging. "
     "Return shipping is free for defective items and paid by the customer otherwise. "
     "Refunds are issued to the original payment method within five business days of inspection."),
    # 2: RoboArm X2 specs
    ("RoboArm X2 Specifications", "product",
     "The RoboArm X2 is a six-axis robotic arm with a payload capacity of five kilograms. "
     "Repeatability is plus or minus zero point zero two millimeters. "
     "It operates on 220 volt power and communicates over Ethernet and Modbus. "
     "Maximum reach is nine hundred millimeters with a top speed of two meters per second."),
    # 3: RoboArm X2 safety
    ("RoboArm X2 Safety Guide", "safety",
     "Operators must complete safety training before using the RoboArm X2. "
     "Keep hands clear of the work envelope during automatic operation. "
     "The emergency stop button halts all motion within two hundred milliseconds. "
     "Light curtains are required for collaborative deployments near human workers."),
    # 4: VisionCam Pro specs
    ("VisionCam Pro Specifications", "product",
     "VisionCam Pro is a machine vision camera with a twelve megapixel sensor. "
     "It captures at sixty frames per second with global shutter. "
     "Lens mount is C-mount and the interface is GigE Vision. "
     "Onboard processing supports defect detection models at the edge."),
    # 5: shipping policy
    ("Shipping Policy", "shipping",
     "Standard shipping takes five to seven business days and costs nine dollars. "
     "Express shipping takes two business days and costs twenty nine dollars. "
     "Orders over five hundred dollars ship free with standard service. "
     "International shipping is available to forty countries with duties calculated at checkout."),
    # 6: data privacy
    ("Data Privacy Notice", "privacy",
     "Acme Robotics collects device telemetry to improve product reliability. "
     "Telemetry data is retained for thirteen months and then anonymized. "
     "Customers may opt out of telemetry collection in the device settings. "
     "Personal data is never sold to third parties."),
    # 7: ConveyorBot 500
    ("ConveyorBot 500 Specifications", "product",
     "ConveyorBot 500 is an autonomous mobile robot for material transport. "
     "Payload capacity is five hundred kilograms with a top speed of one point five meters per second. "
     "Battery life is eight hours with opportunity charging. "
     "Navigation uses lidar SLAM with plus or minus ten millimeter accuracy."),
    # 8: support SLA
    ("Support SLA", "support",
     "Standard support responds within one business day. "
     "Premium support responds within four hours and includes a dedicated engineer. "
     "Critical production-down incidents receive a one hour response under premium plans. "
     "Support hours are Monday to Friday, eight am to six pm Pacific."),
    # 9: firmware updates
    ("Firmware Update Guide", "firmware",
     "Firmware updates are delivered over the air every quarter. "
     "Devices must have at least fifty percent battery before updating. "
     "Do not power off the device during an update. "
     "Rollback to the previous firmware is supported within seventy two hours."),
    # 10: RoboArm X2 maintenance
    ("RoboArm X2 Maintenance", "maintenance",
     "Inspect the RoboArm X2 joints every five hundred operating hours. "
     "Replace the harmonic drive grease every two thousand hours. "
     "Calibrate the tool center point after any collision event. "
     "Maintenance logs must be retained for the life of the product."),
    # 11: payment terms
    ("Payment Terms", "billing",
     "Net thirty payment terms are available for approved business accounts. "
     "All major credit and debit cards are accepted. "
     "Late payments incur a one point five percent monthly finance charge. "
     "Volume discounts start at ten units with five percent off."),
    # 12: VisionCam Pro integration
    ("VisionCam Pro Integration", "integration",
     "VisionCam Pro integrates with the RoboArm X2 over GigE Vision. "
     "Calibration requires the Acme calibration target and takes fifteen minutes. "
     "The SDK supports Python and C++ with sample code included. "
     "ROS 2 drivers are available for Humble and Jazzy distributions."),
    # 13: cybersecurity
    ("Cybersecurity Policy", "security",
     "All Acme devices ship with unique default passwords. "
     "TLS one point three encrypts all cloud communication. "
     "Security patches are released within thirty days of disclosure. "
     "Customers should segment OT networks from corporate IT networks."),
    # 14: ConveyorBot safety
    ("ConveyorBot 500 Safety", "safety",
     "ConveyorBot 500 stops within one meter when an obstacle is detected. "
     "Maximum speed in pedestrian zones is zero point eight meters per second. "
     "Audible warnings sound before the robot resumes motion. "
     "Safety scanners cover a two hundred seventy degree field of view."),
    # 15: training program
    ("Training Program", "training",
     "Operator training is a two day course covering setup and safety. "
     "Advanced programming training is three days and covers the Python SDK. "
     "Certification exams are offered quarterly at regional centers. "
     "On-site training is available for groups of six or more."),
    # 16: SensorHub Mini
    ("SensorHub Mini Specifications", "product",
     "SensorHub Mini aggregates up to sixteen sensor inputs over IO-Link. "
     "It publishes data over MQTT with TLS encryption. "
     "Operating temperature range is minus twenty to sixty degrees Celsius. "
     "Power over Ethernet simplifies installation."),
    # 17: environmental
    ("Environmental Compliance", "compliance",
     "Acme products comply with CE, UL, and FCC regulations. "
     "RoHS compliant materials are used throughout the product line. "
     "Packaging is ninety percent recyclable cardboard. "
     "End-of-life takeback is offered free of charge in the EU."),
    # 18: RoboArm X2 programming
    ("RoboArm X2 Programming", "programming",
     "The RoboArm X2 is programmed via the Acme Studio graphical interface or Python SDK. "
     "Waypoint teaching uses the hand-guide mode with gravity compensation. "
     "Programs are stored as JSON and versioned in the controller. "
     "Simulation mode validates programs before running on hardware."),
    # 19: spare parts
    ("Spare Parts Catalog", "parts",
     "Spare parts ship within three business days from regional warehouses. "
     "Critical spares are stocked for ten years after product discontinuation. "
     "Part numbers follow the format ACM dash four digits. "
     "Warranty replacement parts are shipped express at no charge."),
    # 20: GripForce G3
    ("GripForce G3 Specifications", "product",
     "GripForce G3 is an electric parallel gripper with fifty newton grip force. "
     "Stroke is eighty five millimeters with position feedback. "
     "It mounts directly to the RoboArm X2 wrist flange. "
     "IP54 rating protects against dust and splashing water."),
    # 21: installation
    ("Installation Requirements", "installation",
     "Installation requires a level concrete floor rated for the robot weight. "
     "A dedicated twenty amp circuit is required per robot cell. "
     "Network drops must provide DHCP or static IP configuration. "
     "Ceiling height must exceed the robot reach plus five hundred millimeters."),
    # 22: VisionCam troubleshooting
    ("VisionCam Pro Troubleshooting", "troubleshooting",
     "Blurry images usually indicate a dirty lens or incorrect focus distance. "
     "Dropped frames suggest network congestion on the GigE link. "
     "If the camera is not discovered, check that the firewall allows UDP discovery. "
     "Reset to factory defaults by holding the button for ten seconds."),
    # 23: service contracts
    ("Service Contracts", "service",
     "Annual service contracts include two preventive maintenance visits. "
     "Response time for contract holders is four business hours. "
     "Contracts renew automatically unless cancelled thirty days in advance. "
     "Multi-year contracts receive a ten percent discount."),
    # 24: data export
    ("Data Export Guide", "data",
     "Telemetry data can be exported as CSV or Parquet from the cloud portal. "
     "Exports are limited to one million rows per request. "
     "API access uses OAuth two bearer tokens with hourly refresh. "
     "Webhooks notify on export completion."),
    # 25: RoboArm X2 vs X1
    ("RoboArm X2 vs X1 Comparison", "product",
     "The X2 improves payload from three to five kilograms over the X1. "
     "Repeatability improved from zero point zero five to zero point zero two millimeters. "
     "The X2 adds Modbus support missing from the X1. "
     "X1 programs are forward compatible with the X2 controller."),
    # 26: calibration service
    ("Calibration Service", "service",
     "Factory calibration is included with new RoboArm X2 units. "
     "Recalibration is recommended annually and costs four hundred dollars. "
     "On-site calibration adds travel expenses. "
     "Calibration certificates are ISO 17025 accredited."),
    # 27: SensorHub integration
    ("SensorHub Mini Integration", "integration",
     "SensorHub Mini connects to the Acme cloud over MQTT on port eight eight eight three. "
     "Device provisioning uses X.509 certificates. "
     "Data is buffered locally for up to seven days during outages. "
     "The REST API exposes the latest readings per channel."),
    # 28: emergency procedures
    ("Emergency Procedures", "safety",
     "In an emergency, press the red stop button on any device. "
     "Evacuate the cell and notify the shift supervisor. "
     "Do not restart equipment until the safety officer clears the cell. "
     "Incident reports must be filed within twenty four hours."),
    # 29: software licensing
    ("Software Licensing", "licensing",
     "Acme Studio is licensed per seat with annual renewal. "
     "The Python SDK is free for commercial use. "
     "Floating licenses require a license server on the local network. "
     "Academic licenses are available at fifty percent discount."),
]


@dataclass
class Document:
    doc_id: str
    title: str
    category: str
    text: str


@dataclass
class GoldQuestion:
    question: str
    relevant_doc_ids: list
    expected_keywords: list


def build_corpus() -> list:
    """Return the 30 synthetic documents (deterministic)."""
    return [
        Document(doc_id=f"DOC-{i:02d}", title=title, category=cat, text=text)
        for i, (title, cat, text) in enumerate(_TOPICS)
    ]


def build_gold_questions() -> list:
    """Return the 15 gold questions with relevance judgments."""
    return [
        GoldQuestion(
            "How long does the warranty last and what does it cover?",
            ["DOC-00"], ["twenty four months", "manufacturing defects", "workmanship"]),
        GoldQuestion(
            "What is the return window for a refund?",
            ["DOC-01"], ["thirty days", "full refund"]),
        GoldQuestion(
            "What is the payload capacity of the RoboArm X2?",
            ["DOC-02", "DOC-25"], ["five kilograms", "payload"]),
        GoldQuestion(
            "How fast does the emergency stop halt the RoboArm X2?",
            ["DOC-03"], ["two hundred milliseconds", "emergency stop"]),
        GoldQuestion(
            "What are the VisionCam Pro camera specs?",
            ["DOC-04"], ["twelve megapixel", "sixty frames per second"]),
        GoldQuestion(
            "How much does express shipping cost?",
            ["DOC-05"], ["twenty nine dollars", "two business days"]),
        GoldQuestion(
            "How long is telemetry data retained?",
            ["DOC-06"], ["thirteen months", "anonymized"]),
        GoldQuestion(
            "What is the battery life of the ConveyorBot 500?",
            ["DOC-07"], ["eight hours", "battery"]),
        GoldQuestion(
            "What is the premium support response time?",
            ["DOC-08"], ["four hours", "premium"]),
        GoldQuestion(
            "Can firmware updates be rolled back?",
            ["DOC-09"], ["rollback", "seventy two hours"]),
        GoldQuestion(
            "How often should RoboArm X2 joints be inspected?",
            ["DOC-10"], ["five hundred", "operating hours"]),
        GoldQuestion(
            "What payment terms are offered to businesses?",
            ["DOC-11"], ["net thirty", "business accounts"]),
        GoldQuestion(
            "How does VisionCam Pro connect to the RoboArm X2?",
            ["DOC-04", "DOC-12"], ["GigE Vision", "integrates"]),
        GoldQuestion(
            "What is the grip force of the GripForce G3?",
            ["DOC-20"], ["fifty newton", "grip force"]),
        GoldQuestion(
            "Is the Python SDK free for commercial use?",
            ["DOC-12", "DOC-29"], ["free", "commercial use"]),
    ]
