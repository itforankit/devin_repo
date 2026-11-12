# Kestrel K7 Survey Drone — Operator and Service Manual

Published by Halvorsen Aeronautics, revision 3.2. This manual covers the Kestrel K7
fixed-wing VTOL survey drone, its ground station, batteries, maintenance schedule, warranty,
and support procedures. Read it fully before the first flight.

## 1. Airframe and Specifications

The Kestrel K7 is a hybrid fixed-wing aircraft with four vertical-lift rotors and a single
rear pusher propeller. The wingspan is 2.4 metres and the maximum take-off weight is
9.8 kilograms, including a payload of up to 1.5 kilograms. Cruise speed is 68 km/h and the
maximum speed is 95 km/h. The airframe is made from a carbon-fibre and Kevlar composite and
is rated IP54 against dust and water spray, which means the K7 may fly in light rain but
must never be operated in heavy rain, snow, or hail.

The standard payload bay accepts the Halvorsen Ortho-42 mapping camera (42 megapixels) or
the ThermaScan T2 thermal camera. Swapping payloads takes about two minutes and requires no
tools thanks to the quick-release rail.

## 2. Batteries and Endurance

The K7 uses two hot-swappable 22.2 V lithium-ion packs, model HB-6S-16, each rated at
16,000 mAh. With both packs fully charged and the Ortho-42 camera installed, typical flight
endurance is 110 minutes in calm conditions. Endurance drops to roughly 85 minutes with the
heavier ThermaScan T2, and by a further 15 percent when the wind exceeds 30 km/h.

Charge batteries only with the HC-200 dual-bay charger. A full charge from empty takes
95 minutes. For storage longer than two weeks, packs should be held at a storage charge of
50 to 60 percent and kept between 10 and 25 degrees Celsius. A battery pack is considered
end-of-life after 400 charge cycles or when its measured capacity falls below 80 percent of
the rated value, whichever comes first.

## 3. Ground Station and Communication

The SkyDeck ground station is a ruggedised tablet running the Halvorsen FlightPlan software.
The command-and-control radio link operates at 2.4 GHz with a line-of-sight range of
15 kilometres; live video uses a separate 5.8 GHz link with a range of 8 kilometres.

If the command link is lost for more than 8 seconds, the K7 automatically enters
Return-to-Home (RTH) mode: it climbs to the configured RTH altitude (default 120 metres),
flies back to the take-off point, and lands vertically. If battery charge falls to
20 percent during any mission, the aircraft also triggers RTH regardless of link status.

## 4. Pre-flight Checklist

Before every flight the operator must:

1. Inspect the propellers for chips or cracks; replace any damaged propeller.
2. Confirm both battery packs read at least 95 percent charge.
3. Calibrate the compass if the take-off site is more than 50 kilometres from the previous
   flight location.
4. Verify that the geofence and RTH altitude are set in FlightPlan.
5. Check that the GNSS receiver reports at least 12 satellites before arming.

## 5. Maintenance Schedule

Routine maintenance is based on accumulated flight hours:

- Every 25 flight hours: clean and inspect the motors, check all propeller mounts for
  torque, and update FlightPlan firmware.
- Every 100 flight hours: replace all five propellers and inspect the wing spar joints.
- Every 300 flight hours: send the aircraft to an authorised Halvorsen service centre for a
  full overhaul, which includes motor bearing replacement and a structural X-ray of the spar.

Skipping the 300-hour overhaul voids the airframe warranty.

## 6. Warranty

The airframe and flight controller are covered by a 24-month limited warranty from the date
of purchase. Batteries are covered for 12 months or 200 charge cycles, whichever comes first.
Payload cameras carry their own 12-month warranty. The warranty does not cover crash damage,
water damage from flights in heavy rain, or damage caused by unauthorised modifications.

Customers can purchase the optional KestrelCare Plus plan within 30 days of purchase. It
extends airframe coverage to 36 months and includes one accidental-damage replacement per
year for a fee of 450 euros per incident.

## 7. Support and Repairs

Technical support is available by email at support@halvorsen-aero.example and by phone on
weekdays between 08:00 and 18:00 Central European Time. Priority phone support around the
clock is included only with KestrelCare Plus.

To request a repair, open a Return Merchandise Authorisation (RMA) in the Halvorsen customer
portal and include the aircraft serial number and the flight log of the last mission.
Standard repairs are completed within 10 business days of the aircraft arriving at the
service centre.

## 8. Regulatory Notes

In the European Union the K7 is classified as a C3 class drone. Operators need an EU
Specific-category operational authorisation for flights beyond visual line of sight. The
aircraft broadcasts Remote ID continuously while powered on, and the Remote ID module must
not be disabled.
