from enum import Enum, auto
from typing import Annotated, Tuple

from pydantic import BaseModel

from space_robotics_bench.utils.typing import EnumNameSerializer


class AssetVariant(str, Enum):
    NONE = auto()
    PRIMITIVE = auto()
    DATASET = auto()
    PROCEDURAL = auto()

    def __str__(self) -> str:
        return self.name.lower()


class Domain(str, Enum):
    ASTEROID = auto()
    EARTH = auto()
    MARS = auto()
    MOON = auto()
    ORBIT = auto()

    def __str__(self) -> str:
        return self.name.lower()

    @property
    def gravity_magnitude(self) -> float:
        """
        Magnitude of gravitational acceleration in m/s².

        - Asteroid: 50% gravitational acceleration of Ceres (largest body in the asteroid belt).
        - Orbit: No gravitational acceleration.
        """
        match self:
            case Domain.ASTEROID:
                return 0.14219
            case Domain.EARTH:
                return 9.80665
            case Domain.MARS:
                return 3.72076
            case Domain.MOON:
                return 1.62496
            case Domain.ORBIT:
                return 0.0

    @property
    def gravity_variation(self) -> float:
        """
        Difference between the maximum and minimum of gravitational acceleration in m/s².

        - Asteroid: Ceres is considered as the maximum (largest body in the asteroid belt).
        - Orbit: No gravitational acceleration.
        """
        match self:
            case Domain.ASTEROID:
                return 0.28438
            case Domain.EARTH:
                return 0.0698
            case Domain.MARS:
                return 0.0279
            case Domain.MOON:
                return 0.0253
            case Domain.ORBIT:
                return 0.0

    @property
    def gravity_range(self) -> Tuple[float, float]:
        """
        Range of gravitational acceleration in m/s² calculated as the magnitude ± variation/2.
        """
        magnitude = self.gravity_magnitude()
        delta = self.gravity_variation() / 2.0
        return (magnitude - delta, magnitude + delta)

    @property
    def light_intensity(self) -> float:
        """
        Intensity of Solar light in W/m².

        - Asteroid: Taken at 2.7 AU.
        - Earth | Mars: Taken at the surface. The peak value (sunny day) is subtracted by half of the variation.
        - Moon | Orbit: Taken at 1 AU.
        """
        match self:
            case Domain.ASTEROID:
                return 190.0
            case Domain.EARTH:
                return 775
            case Domain.MARS:
                return 729
            case Domain.MOON | Domain.ORBIT:
                return 1361.0

    @property
    def light_intensity_variation(self) -> float:
        """
        Difference between the maximum and minimum of Solar light intensity in W/m².

        - Asteroid: Approximate range between 2.55 and 2.97 AU.
        - Earth | Mars: Guesstimated effect of atmosphere and weather.
        - Moon | Orbit: Minor variation due to elliptical orbit.
        """
        match self:
            case Domain.ASTEROID:
                return 50.0
            case Domain.EARTH:
                return 450.0
            case Domain.MARS:
                return 226.0
            case Domain.MOON | Domain.ORBIT:
                return 0.5

    @property
    def light_intensity_range(self) -> Tuple[float, float]:
        """
        Range of Solar light intensity in W/m² calculated as the intensity ± variation/2.
        """
        intensity = self.light_intensity()
        delta = self.light_intensity_variation() / 2.0
        return (intensity - delta, intensity + delta)

    @property
    def light_angular_diameter(self) -> float:
        """
        Angular diameter of the Solar light source in degrees.

        - Earth | Mars: Taken at their distance from the Sun.
        - Asteroid | Moon | Orbit: Approximated as a point source due to lack of atmosphere.
        """
        match self:
            case Domain.EARTH:
                return 0.53
            case Domain.MARS:
                return 0.35
            case Domain.ASTEROID | Domain.MOON | Domain.ORBIT:
                return 0.0

    @property
    def light_angular_diameter_variation(self) -> float:
        """
        Variation of the angular diameter of the Solar light source in degrees.
        """
        match self:
            case Domain.EARTH:
                return 0.021
            case Domain.MARS:
                return 0.08
            case Domain.ASTEROID | Domain.MOON | Domain.ORBIT:
                return 0.0

    @property
    def light_angular_diameter_range(self) -> Tuple[float, float]:
        """
        Range of the angular diameter of the Solar light source in degrees calculated as the diameter ± variation/2.
        """
        diameter = self.light_angular_diameter()
        delta = self.light_angular_diameter_variation() / 2.0
        return (diameter - delta, diameter + delta)

    @property
    def light_color_temperature(self) -> float:
        """
        Temperature of the Solar light source in K.

        - Earth | Mars: Guesstimated effect of atmosphere and weather.
        - Asteroid | Moon | Orbit: Intrinsic color temperature of the Sun.
        """
        match self:
            case Domain.EARTH:
                return 5750.0
            case Domain.MARS:
                return 6250.0
            case Domain.ASTEROID | Domain.MOON | Domain.ORBIT:
                return 5778.0

    @property
    def light_color_temperature_variation(self) -> float:
        """
        Variation of the temperature of the Solar light source in K.

        - Earth | Mars: Guesstimated effect of atmosphere and weather.
        - Asteroid | Moon | Orbit: No significant variation.
        """
        match self:
            case Domain.EARTH:
                return 1500.0
            case Domain.MARS:
                return 500.0
            case Domain.ASTEROID | Domain.MOON | Domain.ORBIT:
                return 0.0

    @property
    def light_color_temperature_range(self) -> Tuple[float, float]:
        """
        Range of the temperature of the Solar light source in K calculated as the temperature ± variation/2.
        """

        temperature = self.light_color_temperature()
        delta = self.light_color_temperature_variation() / 2.0
        return (temperature - delta, temperature + delta)


class Asset(BaseModel):
    variant: Annotated[AssetVariant, EnumNameSerializer]


class Assets(BaseModel):
    robot: Asset
    object: Asset
    terrain: Asset
    vehicle: Asset


class EnvironmentCfg(BaseModel):
    domain: Annotated[Domain, EnumNameSerializer]
    assets: Assets
    seed: int
    detail: float
