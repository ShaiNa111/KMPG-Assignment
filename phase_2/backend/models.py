from pydantic import BaseModel, field_validator


class UserInfo(BaseModel):
    full_name: str
    id_number: str
    gender: str
    age: int
    hmo_name: str
    hmo_card_number: str
    membership_tier: str
    is_confirmed: bool = False

    @field_validator('is_confirmed')
    def validate_id_number(cls, v):
        if v is False:
            raise ValueError('User not confirmed')
        return True

    @field_validator('id_number')
    def validate_id(cls, v):
        if v and (not v.isdigit() or len(v) != 9):
            raise ValueError('ID number must be exactly 9 digits')
        return v

    @field_validator('age')
    def validate_age(cls, v):
        if v is not None and (v < 0 or v > 120):
            raise ValueError('Age must be between 0 and 120')
        return v
