# Generated by Django 3.2.2 on 2021-05-17 08:11

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('store', '0006_alter_product_price'),
    ]

    operations = [
        migrations.RenameField(
            model_name='product',
            old_name='savory',
            new_name='kind',
        ),
    ]
