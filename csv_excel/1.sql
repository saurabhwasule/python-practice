SELECT distinct source,heading,price,super_buildup,owner_dealer_name AS name 
FROM PROPERTY_DETAIL1
where 1=1 
AND REPLACE(SUBSTRING_INDEX(heading, ',', 1), 'Bedroom', 'BHK')='1 BHK'
AND location='Bellandur'
AND owner_dealer='Owner';