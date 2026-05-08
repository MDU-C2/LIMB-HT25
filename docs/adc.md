# ADC on the ESP32-C3-Zero

The way the ADC works on the ESP32-C3-Zero is that the voltage
$V_\text{in}$ fed to the ADC is linearly interpolated from the
range $`\left[0, V_\text{ref}\right]`$ onto the range
$`\left[0, 2^\text{bitwidth} - 1\right]`$, where $V_\text{ref}$ is the ADC's
internal reference voltage and bitwidth determines the value that is returned from the ADC
when $V_\text{in} = V_\text{ref}$. The equation governing this linear interpolation
is shown
[in the documentation](https://docs.espressif.com/projects/esp-idf/en/v6.0.1/esp32c3/api-reference/peripherals/adc/index.html#adc-conversion),
with $V_\text{data}$ representing $V_\text{in}$.

$V_\text{ref}$ is designed to be $1.1 \mathrm{V}$ for the ESP32-C3-Zero, meaning that the maximum voltage that can be measured using the ADC
is $1.1 \mathrm{V}$. However, the ADC can be configured to extend the input voltage range by setting an attenuation level.
The maximum voltage that can be read is then increased by dividing $V_\text{ref}$ by a coefficient $k$ corresponding
to the selected attenuation level. Using the maximum attenuation level of $12\ \text{dB}$,
$k \approx 0.25$ ([from the Technical Reference Manual](https://documentation.espressif.com/esp32-c3_technical_reference_manual_en.pdf#subsubsection.34.2.3.2)),
meaning that the maximum voltage that can be read is $V_\text{ref} / 0.25 = 1.1 \mathrm{V} / 0.25 = 4.4 \mathrm{V}$.

However, although the ADC can measure voltages up to $4.4 \mathrm{V}$, values at
the ends of the measurement range do not map onto the voltage linearly. As such, the
effective measurement range that stays linear for an attenuation level
of $12\ \text{dB}$ is about $`\left[0, 2.5\right] \mathrm{V}`$
([Table 5-6 in the C3's datasheet](https://documentation.espressif.com/esp32-c3_datasheet_en.pdf#subsection.5.5)).

To get the best results from the ADC measurements, we then want to make sure that any voltage passed to
the ADC is below $2.5 \mathrm{V}$. For the sensors that provide voltages up to $3.3 \mathrm{V}$, we use voltage dividers
with $R_1 = 1\ \text{k}\Omega$ and $R_2 = 2\ \text{k}\Omega$ such that the voltage is stepped down to

$$
V_\text{out} = 3.3 \mathrm{V} \times \frac{2\ \text{k}\Omega}{1\ \text{k}\Omega + 2\ \text{k}\Omega} = 2.2 \mathrm{V}.
$$

## More information
More infomation can be found in
[the datasheet](https://documentation.espressif.com/esp32-c3_datasheet_en.pdf#subsection.5.5),
[the Technical Reference Manual](https://documentation.espressif.com/esp32-c3_technical_reference_manual_en.pdf#subsubsection.34.2.3.2),
and
[the documentation](https://docs.espressif.com/projects/esp-idf/en/v6.0.1/esp32c3/api-reference/peripherals/adc/index.html).

